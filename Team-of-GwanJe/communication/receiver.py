from struct import unpack, calcsize
from serial import Serial, PARITY_NONE, STOPBITS_ONE, EIGHTBITS   # 8N1
from threading import Thread
from time import sleep, perf_counter

import numpy as np

# ---- 프로토콜 합의 ----
HEAD = b'AB'
TAIL = b'Z'
# CHANGED: 16 * float64('d') 로 변경, 끝에 u8 체크섬
FMT_BODY = '<BBBB16dB'                         # 시간4 + 16*float64 + u8 checksum
BODY_SIZE = calcsize(FMT_BODY)                 # CHANGED: 133
HEADER_SIZE = 3                                # 'A','B', len
PACKET_SIZE = HEADER_SIZE + BODY_SIZE + 1      # CHANGED: + 'Z' = 137

#패킷 'A', 'B', len, h, m, s, tenms,
#     est_x, est_y, est_z,
#     est_vx, est_vy, est_vz,
#     a_x, a_y, a_z,
#     q0, q1, q2, q3,
#     w_x, w_y, w_z,
#     checksum, 'Z'

class Receiver(Thread):
    def __init__(self, datahub):
        super().__init__(daemon=True)
        self.datahub = datahub
        self.ser = None
        self.first_time = True

        # 상태/진단용
        self.good_frames = 0
        self.bad_streak  = 0
        self.last_byte_ts = None

        # 타임아웃/오류 판정
        self.ERR_THRESHOLD  = 20
        self.NO_BYTE_TIMEOUT = 0.8  # s

    def setSerial(self, myport, mybaudrate):
        self.ser = Serial(
            port=myport,
            baudrate=int(mybaudrate),
            parity=PARITY_NONE,
            stopbits=STOPBITS_ONE,     # 8N1
            bytesize=EIGHTBITS,
            timeout=0.1,
        )

    def _read_exact(self, n: int):
        """정확히 n바이트 읽기 (타임아웃 시 None)."""
        buf = bytearray()
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
            # 바이트가 들어온 시점 갱신
            self.last_byte_ts = perf_counter()
        return bytes(buf)

    def _parse_body(self, body: bytes) -> bool:
        """본문(133B) 파싱 + u8 체크섬 검증 후 datahub에 반영."""
        if len(body) != BODY_SIZE:
            return False

        # CHANGED: u8 checksum over (time4 + 16*float64) == body[:-1] (총 132B)
        cs_calc = sum(body[:-1]) & 0xFF
        cs_rx   = body[-1]
        if cs_calc != cs_rx:
            return False

        # unpack: h,m,s,tenms, (16 doubles), cs
        tup = unpack(FMT_BODY, body)
        h, m, s, tenms = tup[:4]
        floats16       = tup[4:-1]   # 16개 double
        # cs = tup[-1]  # 이미 위에서 검증 완료

        # CHANGED: 기존 update 시그니처 유지 위해 float32로 캐스팅해서 전달
        out = np.array([h, m, s, tenms, *floats16], dtype=np.float32)
        self.datahub.update(out)
        return True

    def run(self):
        hdr_stage = 0  # 0: find 'A', 1: need 'B', 2: read len
        while True:
            try:
                if self.datahub.iscommunication_start:
                    # 포트 오픈/초기화
                    if self.first_time:
                        self.setSerial(self.datahub.mySerialPort, self.datahub.myBaudrate)
                        self.first_time = False
                    if not self.ser.is_open:
                        self.ser.open()
                        self.ser.reset_input_buffer()
                        self.ser.reset_output_buffer()
                        self.good_frames = 0
                        self.bad_streak  = 0
                        self.datahub.serial_port_error = -1
                        self.last_byte_ts = perf_counter()

                    b = self.ser.read(1)
                    if not b:
                        # 아무 바이트도 안 들어오고 최초 동기화도 못했을 때만 타임아웃 에러
                        if (self.good_frames == 0 and
                            self.last_byte_ts and
                            (perf_counter() - self.last_byte_ts) > self.NO_BYTE_TIMEOUT):
                            self.datahub.serial_port_error = 1
                        continue

                    self.last_byte_ts = perf_counter()

                    if hdr_stage == 0:
                        hdr_stage = 1 if b == b'A' else 0
                    elif hdr_stage == 1:
                        hdr_stage = 2 if b == b'B' else 0
                    elif hdr_stage == 2:
                        msg_len = b[0]
                        # CHANGED: LEN은 본문 길이(=133)여야 함
                        if msg_len != BODY_SIZE:
                            hdr_stage = 0
                            # 동기화 전이면 bad_streak 카운트
                            if self.good_frames == 0:
                                self.bad_streak += 1
                                if self.bad_streak >= self.ERR_THRESHOLD:
                                    self.datahub.serial_port_error = 1
                            continue

                        # 본문 133바이트
                        body = self._read_exact(msg_len)
                        # 테일 1바이트
                        tail = self._read_exact(1)
                        hdr_stage = 0

                        if (body is not None) and (tail == TAIL) and self._parse_body(body):
                            self.good_frames += 1
                            self.bad_streak = 0
                            if self.good_frames == 1:
                                self.datahub.serial_port_error = 0
                        else:
                            if self.good_frames == 0:
                                self.bad_streak += 1
                                if self.bad_streak >= self.ERR_THRESHOLD:
                                    self.datahub.serial_port_error = 1

                else:
                    # 통신 중지 상태
                    if self.ser is not None and self.ser.is_open:
                        self.ser.close()
                    self.datahub.serial_port_error = -1
                    sleep(0.05)

            except Exception:
                # 예외 발생 시 재동기화
                self.datahub.serial_port_error = 1
                hdr_stage = 0
                sleep(0.05)

# 단독 실행 테스트용 (통합 환경에선 사용 안 해도 됨)
if __name__ == "__main__":
    try:
        from datahub import Datahub
        dh = Datahub()
        dh.iscommunication_start = True
        r = Receiver(dh)
        r.start()
        r.join()
    except Exception as e:
        #print("Receiver self-test failed:", e)
        pass
