from struct import pack
from serial import Serial, PARITY_NONE, STOPBITS_ONE, EIGHTBITS
from threading import Thread, Lock
from queue import Queue, Empty
from time import sleep
import numpy as np
from typing import Sequence, Union, Iterable, Callable, Optional


class Sender(Thread):
    """
    - 기본 동작: Receiver가 연 시리얼 핸들을 공유(get_serial로 주입)하여 write만 수행.
      -> 동일 포트를 2번 여는 문제 방지.
    - get_serial을 주지 않으면 setSerial(...)로 직접 열어 사용할 수도 있음(8N1).
    - 전송 실패 플래그는 datahub.sender_error (-1:대기, 0:성공/정상, 1:실패) 만 사용.
      수신 쪽 datahub.serial_port_error 는 건드리지 않음.
    """
    def __init__(self,
                 datahub=None,
                 *,
                 get_serial: Optional[Callable[[], Optional[Serial]]] = None,
                 serial_lock: Optional[Lock] = None,
                 endianness: str = '<'):
        super().__init__(daemon=True)
        self.datahub = datahub
        self.endianness = endianness  # '<' little-endian (default), '>' big-endian

        # 공유 모드: Receiver가 가진 ser을 콜백으로 받아옴
        self._get_serial = get_serial
        self._shared_lock = serial_lock or Lock()

        # 독립 모드: Sender가 직접 open한 ser을 보관
        self._own_ser: Optional[Serial] = None
        self._first_time = True

        self._q: "Queue[tuple[str, object]]" = Queue()
        self._running = True

        if self.datahub is not None:
            # 송신 상태 플래그 (수신 플래그와 분리)
            if not hasattr(self.datahub, "sender_error"):
                self.datahub.sender_error = -1

    # ---------- Serial (독립 모드에서만 사용) ----------
    def setSerial(self, port: str, baudrate: int):
        """독립 모드: Sender가 직접 포트를 열어 사용(8N1). 가능하면 공유 모드 권장."""
        self._own_ser = Serial(
            port=port,
            baudrate=int(baudrate),
            parity=PARITY_NONE,
            stopbits=STOPBITS_ONE,      # MCU/Receiver와 동일(8N1)
            bytesize=EIGHTBITS,
            timeout=0.2,
        )

    def _current_ser(self) -> Optional[Serial]:
        """현재 사용할 Serial 핸들(공유 우선)."""
        if self._get_serial is not None:
            try:
                return self._get_serial()
            except Exception:
                return None
        return self._own_ser

    def _ensure_open_if_own(self):
        """독립 모드일 때만 open 보장. 공유 모드에선 Receiver가 관리."""
        if self._get_serial is not None:
            return  # 공유 모드: Receiver가 열고 닫음
        if self._own_ser is None:
            raise RuntimeError("Serial not initialized; call setSerial(...) first or provide get_serial.")
        if not self._own_ser.is_open:
            self._own_ser.open()

    # ========== 버튼 패킷 ==========
    def build_packet_button(self, button_byte: int) -> bytes:
        """
        Packet: 'A','B', data_len, button_data, checksum, 'Z'
        - data_len: 3 (button + checksum + 'Z')
        - checksum: 여기선 button 바이트를 그대로 사용 (필요 시 XOR/합 등으로 교체 가능)
        """
        b = int(button_byte) & 0xFF
        checksum = b
        payload = bytes([b, checksum]) + b'Z'      # 3 bytes
        header  = b'A' + b'B' + bytes([len(payload)])
        return header + payload

    def send_button_now(self, button_byte: int) -> bool:
        """버튼 바이트 1개를 즉시(동기) 전송."""
        try:
            self._ensure_open_if_own()
            ser = self._current_ser()
            if ser is None or not ser.is_open:
                raise RuntimeError("Serial not available/open.")
            pkt = self.build_packet_button(button_byte)
            with self._shared_lock:
                ser.write(pkt)
            if self.datahub is not None:
                self.datahub.sender_error = 0
            return True
        except Exception:
            if self.datahub is not None:
                self.datahub.sender_error = 1
            return False

    def enqueue_button(self, button_byte: int):
        self._q.put(("button", int(button_byte) & 0xFF))

    # ========== float 패킷(기존 호환) ==========
    def build_packet(self, order_values: Sequence[Union[float, int]]) -> bytes:
        arr = np.asarray(order_values, dtype=np.float32).ravel()
        if arr.size == 0:
            raise ValueError("order_values must contain at least one number.")

        checksum = np.float32(np.sum(arr))
        body = pack(f'{self.endianness}{arr.size}f', *arr) + pack(f'{self.endianness}f', float(checksum))
        payload = body + b'Z'
        msg_len = len(payload)
        if msg_len > 255:
            # 4*N + 4 + 1 <= 255 → N <= 62
            raise ValueError(f"Payload too large ({msg_len} bytes). Reduce number of floats.")
        header = b'A' + b'B' + bytes([msg_len])
        return header + payload

    def send_now(self, order_values: Sequence[Union[float, int]]) -> bool:
        try:
            self._ensure_open_if_own()
            ser = self._current_ser()
            if ser is None or not ser.is_open:
                raise RuntimeError("Serial not available/open.")
            pkt = self.build_packet(order_values)
            with self._shared_lock:
                ser.write(pkt)
            if self.datahub is not None:
                self.datahub.sender_error = 0
            return True
        except Exception:
            if self.datahub is not None:
                self.datahub.sender_error = 1
            return False

    # ---------- 큐 전송 ----------
    def enqueue(self, order_values: Union[float, int, Iterable[Union[float, int]]]):
        if isinstance(order_values, (float, int)):
            vals = [order_values]
        else:
            vals = list(order_values)
        if not vals:
            raise ValueError("enqueue requires at least one value.")
        self._q.put(("floats", vals))

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                # 통신 On/Off는 datahub 플래그를 따름(없으면 항상 On)
                comm_on = bool(getattr(self.datahub, "iscommunication_start", True))
                if not comm_on:
                    sleep(0.05)
                    continue

                # 독립 모드일 때만 open 보장
                if self._first_time:
                    if self._get_serial is None and self._own_ser is None:
                        # 독립 모드인데 setSerial 안 했으면 구성 오류
                        raise RuntimeError("Sender: setSerial(...) or get_serial must be provided.")
                    self._first_time = False
                self._ensure_open_if_own()

                # 큐에서 하나 가져와 전송
                try:
                    kind, payload = self._q.get(timeout=0.05)
                except Empty:
                    continue

                ser = self._current_ser()
                if ser is None or not ser.is_open:
                    # 공유 모드: 아직 Receiver가 포트를 못 열었을 수 있음
                    continue

                if kind == "button":
                    pkt = self.build_packet_button(int(payload))
                else:  # "floats"
                    pkt = self.build_packet(payload)

                with self._shared_lock:
                    ser.write(pkt)

                if self.datahub is not None:
                    self.datahub.sender_error = 0

            except Exception:
                if self.datahub is not None:
                    self.datahub.sender_error = 1
                sleep(0.05)

        # 종료 시: 독립 모드면 닫고, 공유 모드면 건드리지 않음
        try:
            if self._get_serial is None and self._own_ser and self._own_ser.is_open:
                self._own_ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    import time

    # 예시 1) 독립 모드 (테스트 장비로 송신만)
    # snd = Sender(get_serial=None)
    # snd.setSerial(port='COM3', baudrate=115200)
    # snd.start()
    # snd.enqueue_button(0x91)
    # snd.enqueue([1.0, 2.0, 3.5])
    # time.sleep(0.5)
    # snd.stop(); snd.join()

    # 예시 2) 공유 모드 (Receiver가 연 포트를 공유)
    #   from communication.receiver import Receiver
    #   rx = Receiver(datahub); rx.start()
    #   snd = Sender(datahub, get_serial=lambda: rx.ser)
    #   snd.start()
    pass
