# sender.py
from serial import Serial, PARITY_NONE, STOPBITS_ONE, EIGHTBITS  # ← 8N1로 맞춤
from threading import Thread, Lock
from queue import Queue, Empty
from time import sleep
from typing import Optional, Callable, Tuple
import time  # ACK 대기용

def bin8_with_space(val: int) -> str:
    s = format(int(val) & 0xFF, '08b')
    return f"{s[:4]} {s[4:]}"

class Sender(Thread):
    """
    버튼 전용 패킷 송신 스레드 (Receiver와 다른 포트를 사용).
      패킷: 'A','B', len, button_data, checksum, 'Z'
        - len = 3
        - checksum = button_data 그대로
    포트/보드레이트는 datahub.mySendSerialPort / datahub.mySendBaudrate(문자열 허용)에서 자동 반영.
    datahub.sender_error: -1(대기), 0(정상), 1(실패)
    """
    def __init__(self,
                 datahub=None,
                 *,
                 get_serial: Optional[Callable[[], Optional[Serial]]] = None,
                 serial_lock: Optional[Lock] = None):
        super().__init__(daemon=True)
        self.datahub = datahub

        self._get_serial = get_serial
        self._shared_lock = serial_lock or Lock()

        self._own_ser: Optional[Serial] = None
        self._first_time = True
        self._cfg_port: Optional[str] = None
        self._cfg_baud: Optional[int] = None

        self._q: "Queue[Tuple[str, int]]" = Queue()
        self._running = True

        # ACK 검증 설정
        self.verify_ack = True       # 시나리오 A: True
        self.ack_timeout = 0.25      # 250 ms 내에 ACK 수신 기대
        self.ack_chunk = 64          # read() 청크 크기

        if self.datahub is not None and not hasattr(self.datahub, "sender_error"):
            self.datahub.sender_error = -1

    # ---------- Serial (독립 모드) ----------
    def setSerial(self,
                  port: str,
                  baudrate: int,
                  *,
                  parity=PARITY_NONE,
                  stopbits=STOPBITS_ONE,   # MCU와 일치(8N1)
                  bytesize=EIGHTBITS,
                  timeout=0.2,
                  write_timeout=0.2):
        """독립 모드: Sender가 직접 포트를 열어 사용."""
        self._own_ser = Serial(
            port=port,
            baudrate=int(baudrate),
            parity=parity,
            stopbits=stopbits,
            bytesize=bytesize,
            timeout=timeout,
            write_timeout=write_timeout,
        )
        self._cfg_port, self._cfg_baud = port, int(baudrate)
        print(f"[Sender] Opened port={port}, baud={baudrate}, stopbits={stopbits}")

    def switch_port(self, port: str, baudrate: int, **kwargs):
        """런타임에 포트를 변경."""
        with self._shared_lock:
            try:
                if self._own_ser and self._own_ser.is_open:
                    self._own_ser.close()
            except Exception:
                pass
            self.setSerial(port, baudrate, **kwargs)

    def _current_ser(self) -> Optional[Serial]:
        """현재 사용 Serial 핸들(공유 우선. 독립 모드 권장)."""
        if self._get_serial is not None:
            try:
                return self._get_serial()
            except Exception:
                return None
        return self._own_ser

    def _ensure_open_if_own(self):
        """독립 모드일 때만 open 보장. 공유 모드에선 Receiver가 관리."""
        if self._get_serial is not None:
            return
        if self._own_ser is None:
            raise RuntimeError("Serial not initialized; call setSerial(...) first.")
        if not self._own_ser.is_open:
            self._own_ser.open()

    # ---------- Datahub에서 포트/보드레이트 가져오기 ----------
    def _parse_baud(self, val) -> int:
        try:
            s = str(val).strip()
            return int(s) if s else 115200
        except Exception:
            return 115200

    def _apply_cfg_from_datahub(self):
        """datahub.mySendSerialPort / mySendBaudrate를 읽어 포트를 열거나 변경."""
        if self._get_serial is not None:
            return  # 공유 모드면 무시
        if self.datahub is None:
            return

        port = getattr(self.datahub, "mySendSerialPort", None)
        baud = self._parse_baud(getattr(self.datahub, "mySendBaudrate", None))

        if not port:
            return  # 아직 UI에서 입력 안 됨

        # 최초 오픈
        if self._own_ser is None:
            self.setSerial(port, baud)
            return

        # 변경 감지 시 재오픈
        if (self._cfg_port != port) or (self._cfg_baud != baud):
            print(f"[Sender] Port config changed: {self._cfg_port}->{port}, {self._cfg_baud}->{baud}")
            self.switch_port(port, baud)

    # ========== 버튼 패킷 ==========
    @staticmethod
    def build_packet_button(button_byte: int) -> bytes:
        """
        Packet: 'A','B', data_len, button_data, checksum, 'Z'
        - data_len = 3
        - checksum = button_data 그대로
        """
        b = int(button_byte) & 0xFF
        payload = bytes([b, b]) + b'Z'                 # [button, checksum(=button), 'Z']
        header  = b'A' + b'B' + bytes([len(payload)])  # len == 3
        return header + payload

    # ---------- ACK 대기/검증 ----------
    def _wait_for_ack(self, ser: Serial, expected_btn: int) -> Tuple[bool, Optional[int], bytes]:
        """
        ACK 형태: b'OK' + <btn_byte> + '\\r\\n' (총 5바이트).
        스트림 중간 어디에 있어도 'OK' 토큰을 찾아 해석.
        반환: (성공여부, echo_btn or None, 수집버퍼)
        """
        deadline = time.perf_counter() + float(self.ack_timeout)
        buf = bytearray()
        old_timeout = ser.timeout
        try:
            ser.timeout = 0.05  # 짧은 폴링
            while time.perf_counter() < deadline:
                chunk = ser.read(self.ack_chunk)
                if chunk:
                    buf.extend(chunk)
                    i = buf.find(b'OK')
                    if i != -1:
                        if i + 2 < len(buf):  # 'OK' 다음 바이트가 도착했는가?
                            echo_btn = buf[i + 2]
                            # CRLF는 선택적 — 굳이 검사 안 해도 됨
                            return (echo_btn == (expected_btn & 0xFF), echo_btn, bytes(buf))
                else:
                    # 데이터 없음 — 계속 대기
                    pass
            # 타임아웃
            return (False, None, bytes(buf))
        finally:
            ser.timeout = old_timeout

    def _send_and_verify(self, ser: Serial, pkt: bytes, btn: int) -> bool:
        with self._shared_lock:
            n = ser.write(pkt)
        print(f"[Sender] TX button={bin8_with_space(btn)} (0x{btn:02X}), bytes={n}")

        if not self.verify_ack:
            if self.datahub is not None:
                self.datahub.sender_error = 0
            return True

        ok, echo, raw = self._wait_for_ack(ser, btn)
        if ok:
            print(f"[Sender] ACK OK (btn=0x{echo:02X})")
            if self.datahub is not None:
                self.datahub.sender_error = 0
            return True
        else:
            print(f"[Sender] ACK FAIL (echo={echo}, raw={raw})")
            if self.datahub is not None:
                self.datahub.sender_error = 1
            return False

    def send_button_now(self, button_byte: int) -> bool:
        """버튼 바이트 1개를 즉시(동기) 전송 + (옵션)ACK 검증."""
        try:
            self._apply_cfg_from_datahub()
            self._ensure_open_if_own()

            ser = self._current_ser()
            if ser is None or not ser.is_open:
                raise RuntimeError("Serial not available/open.")

            pkt = self.build_packet_button(button_byte)
            return self._send_and_verify(ser, pkt, button_byte)
        except Exception as e:
            print(f"[Sender] send_button_now error: {e}")
            if self.datahub is not None:
                self.datahub.sender_error = 1
            return False

    # ---------- 큐 API ----------
    def enqueue_button(self, button_byte: int):
        """버튼 바이트 전송 요청을 큐에 등록."""
        self._q.put(("button", int(button_byte) & 0xFF))

    def stop(self):
        self._running = False

    # ---------- 스레드 루프 ----------
    def run(self):
        while self._running:
            try:
                # UI 저장값 반영(포트/보드레이트 자동 오픈·변경)
                self._apply_cfg_from_datahub()

                # 통신 On/Off 플래그(없으면 항상 On)
                comm_on = bool(getattr(self.datahub, "iscommunication_start", True))
                if not comm_on:
                    sleep(0.05)
                    continue

                # 초기 구성 확인
                if self._first_time:
                    if self._get_serial is None and self._own_ser is None:
                        sleep(0.05)
                        continue
                    self._first_time = False

                # 독립 모드면 open 보장
                self._ensure_open_if_own()

                # 큐에서 하나 가져와 전송
                try:
                    kind, payload = self._q.get(timeout=0.05)
                except Empty:
                    continue

                ser = self._current_ser()
                if ser is None or not ser.is_open:
                    continue

                if kind == "button":
                    pkt = self.build_packet_button(payload)
                else:
                    continue  # 현재는 버튼 패킷만

                self._send_and_verify(ser, pkt, payload)

            except Exception as e:
                print(f"[Sender] run loop error: {e}")
                if self.datahub is not None:
                    self.datahub.sender_error = 1
                sleep(0.05)

        # 종료 처리: 독립 모드면 닫기
        try:
            if self._get_serial is None and self._own_ser and self._own_ser.is_open:
                self._own_ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    # 간이 테스트: datahub 흉내
    class Dummy:
        mySendSerialPort = "COM16"
        mySendBaudrate   = "115200"
        iscommunication_start = True

    import time
    dh = Dummy()
    snd = Sender(dh)           # setSerial 호출 필요 없음 — datahub 값으로 자동 오픈
    snd.start()
    snd.enqueue_button(0x91)   # 예: 1001 0001
    time.sleep(0.3)
    snd.stop(); snd.join()