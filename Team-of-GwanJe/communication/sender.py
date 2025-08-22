from struct import pack
from serial import Serial, PARITY_NONE, STOPBITS_TWO, EIGHTBITS
from threading import Thread, Lock
from queue import Queue, Empty
from time import sleep
import numpy as np
from typing import Sequence, Union, Iterable

class Sender(Thread):
    def __init__(self, datahub=None, *, endianness: str = '<'):
        super().__init__(daemon=True)
        self.datahub = datahub
        self.endianness = endianness  # '<' little-endian (default), '>' big-endian
        self.ser: Serial | None = None
        self.first_time = True
        self._lock = Lock()
        self._q: "Queue[tuple[str, object]]" = Queue()  # (kind, payload)
        self._running = True

        # Optional: mirror Receiver.py's error flag if datahub provided
        if self.datahub is not None and not hasattr(self.datahub, "serial_port_error"):
            self.datahub.serial_port_error = 0

    # ---------- Serial ----------
    def setSerial(self, port: str, baudrate: int):
        self.ser = Serial(
            port=port,
            baudrate=int(baudrate),
            parity=PARITY_NONE,
            stopbits=STOPBITS_TWO,
            bytesize=EIGHTBITS,
            timeout=0.2,
        )

    def _ensure_open(self):
        if self.ser is None:
            # Prefer Datahub config if available
            if self.datahub is None:
                raise RuntimeError("Serial not initialized; call setSerial(...) first.")
            self.setSerial(self.datahub.mySerialPort, self.datahub.myBaudrate)
        if not self.ser.is_open:
            self.ser.open()

    # ========== 신규: 버튼 패킷 ==========
    def build_packet_button(self, button_byte: int) -> bytes:
        """
        Packet: 'A','B', data_len, button_data, checksum, 'Z'
        - data_len: payload 길이 (button_data + checksum + 'Z') = 3
        - checksum: 간단히 button_data의 8비트 합 (여기선 그대로 에코)로 설정
                    필요 시 (button_data) & 0xFF, 또는 (~sum)&0xFF 등으로 변경 가능
        """
        b = int(button_byte) & 0xFF
        checksum = b  # <-- 필요하면 (b) & 0xFF, 또는 (~b)&0xFF 등으로 바꿔도 됨

        payload = bytes([b, checksum]) + b'Z'  # [button_data, checksum, 'Z']
        data_len = len(payload)                # 3
        if data_len > 255:
            raise ValueError("Payload too large.")

        header = b'A' + b'B' + bytes([data_len])
        return header + payload

    def send_button_now(self, button_byte: int) -> bool:
        """버튼 바이트 1개를 즉시(동기) 전송."""
        try:
            self._ensure_open()
            pkt = self.build_packet_button(button_byte)
            with self._lock:
                self.ser.write(pkt)
            if self.datahub is not None:
                self.datahub.serial_port_error = 0
            return True
        except Exception:
            if self.datahub is not None:
                self.datahub.serial_port_error = 1
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass
            return False

    def enqueue_button(self, button_byte: int):
        """스레드 루프에서 보낼 버튼 패킷을 큐에 적재."""
        self._q.put(("button", int(button_byte) & 0xFF))

    # ========== 기존: float 패킷(유지) ==========
    def build_packet(self, order_values: Sequence[Union[float, int]]) -> bytes:
        # Normalize to float32 array
        arr = np.asarray(order_values, dtype=np.float32).ravel()
        if arr.size == 0:
            raise ValueError("order_values must contain at least one number.")

        # checksum as float32 sum
        checksum = np.float32(np.sum(arr))

        # body = order floats + checksum (float32)
        body = pack(f'{self.endianness}{arr.size}f', *arr) + pack(f'{self.endianness}f', float(checksum))

        # payload = body + 'Z'
        payload = body + b'Z'
        msg_len = len(payload)  # must fit in one byte
        if msg_len > 255:
            # Max floats N such that 4*N + 4 + 1 <= 255 -> N <= 62
            raise ValueError(f"Payload too large ({msg_len} bytes). Reduce number of floats.")

        # header = 'A','B',len
        header = b'A' + b'B' + bytes([msg_len])

        return header + payload

    def send_now(self, order_values: Sequence[Union[float, int]]) -> bool:
        """기존 float 패킷 동기 전송 (호환 유지)."""
        try:
            self._ensure_open()
            pkt = self.build_packet(order_values)
            with self._lock:
                self.ser.write(pkt)
            if self.datahub is not None:
                self.datahub.serial_port_error = 0
            return True
        except Exception:
            if self.datahub is not None:
                self.datahub.serial_port_error = 1
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass
            return False

    # ---------- Queued / threaded send ----------
    def enqueue(self, order_values: Union[float, int, Iterable[Union[float, int]]]):
        """기존 float 패킷 큐잉 (호환 유지)."""
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
        """
        datahub가 있으면:
        - datahub.iscommunication_start 가 True일 때 포트 열고 큐에서 전송
        - False면 닫고 대기
        """
        while self._running:
            try:
                comm_on = True
                if self.datahub is not None:
                    comm_on = bool(self.datahub.iscommunication_start)

                if comm_on:
                    if self.first_time:
                        if self.ser is None:
                            if self.datahub is None:
                                raise RuntimeError("Serial not initialized; call setSerial(...) or provide datahub.")
                            self.setSerial(self.datahub.mySerialPort, self.datahub.myBaudrate)
                        self.first_time = False

                    self._ensure_open()
                    if self.datahub is not None:
                        self.datahub.serial_port_error = 0

                    try:
                        kind, payload = self._q.get(timeout=0.05)
                    except Empty:
                        continue

                    if kind == "button":
                        pkt = self.build_packet_button(int(payload))
                    else:  # "floats"
                        pkt = self.build_packet(payload)

                    with self._lock:
                        self.ser.write(pkt)

                else:
                    if self.ser is not None and self.ser.is_open:
                        self.ser.close()
                    sleep(0.05)

            except Exception:
                if self.datahub is not None:
                    self.datahub.serial_port_error = 1
                sleep(0.05)

        # cleanup on stop
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    import time

    snd = Sender(datahub=None, endianness='<')
    snd.setSerial(port='COM3', baudrate=9600)
    snd.start()

    # 버튼 전송 (예: 0b10010001)
    snd.enqueue_button(0x91)

    # 기존 float 전송도 그대로 가능
    snd.enqueue([1.0, 2.0, 3.5])

    time.sleep(0.5)
    snd.stop()
    snd.join()
