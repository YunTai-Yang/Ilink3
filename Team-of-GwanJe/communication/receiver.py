from struct import unpack, calcsize
from serial import Serial, PARITY_NONE, STOPBITS_TWO, EIGHTBITS
from threading import Thread
from time import sleep
import numpy as np


# 패킷: 'A','B', len, hours, mins, secs, tenmilis,
#       E, N, U, v_E, v_N, v_U, a_p, a_y, a_r, q_0, q_1, q_2, q_3, w_p, w_y, w_r,
#       checksum, 'Z'

##'A', 'B', len, order, checksum, 'Z' 

class Receiver(Thread):
    def __init__(self, datahub):
        super().__init__(daemon=True)
        self.datahub = datahub
        self.first_time = True
        self.ser = None

        # 본문 포맷 (마지막 'Z' 제외): 4바이트 시간 + 16개 float + checksum(float)
        self.DATA_FMT_BODY = '<BBBB16ff'
        self.DATA_BODY_SIZE = calcsize(self.DATA_FMT_BODY)  # 4 + 16*4 + 4 = 72 bytes
        self.DATA_PAYLOAD_SIZE = self.DATA_BODY_SIZE + 1    # + 'Z' 1바이트 = 73 bytes

    def setSerial(self, myport, mybaudrate):
        self.ser = Serial(
            port=myport,
            baudrate=int(mybaudrate),
            parity=PARITY_NONE,
            stopbits=STOPBITS_TWO,
            bytesize=EIGHTBITS,
            timeout=0.1,
        )

    def _parse_and_update(self, payload: bytes):
        """
        payload: len 바이트로 읽은 '본문 + Z'
        반환: True(성공) / False(실패)
        """
        if len(payload) != self.DATA_PAYLOAD_SIZE:
            return False

        # 트레일러 확인
        if payload[-1] != ord('Z'):
            return False

        body = payload[:-1]  # 'Z' 제외

        try:
            # 언팩: (h, m, s, tm, 16 floats..., checksum)
            unpacked = unpack(self.DATA_FMT_BODY, body)

            hours, mins, secs, tenmilis = unpacked[:4]
            floats16 = unpacked[4:-1]   # E..q3 (16개)
            checksum = unpacked[-1]     # float checksum

            #float32 기준 체크섬
            calc = np.float32(hours + mins + secs + tenmilis + np.sum(np.array(floats16, dtype=np.float32)))

            # 허용 오차 내 비교 (센서 부동소수 오차 여지)
            if not np.isfinite(checksum) or abs(np.float32(checksum) - calc) > 1e-3:
                return False

            # Datahub에 넘길 배열 구성 (시간4 + 16 floats)
            # 순서: hours, mins, secs, tenmilis,
            #       E, N, U, v_E, v_N, v_U, a_p, a_y, a_r, w_p, w_y, w_r, q_0, q_1, q_2, q_3
            out = np.array([hours, mins, secs, tenmilis, *floats16], dtype=np.float32)

            # 기존 인터페이스 유지
            self.datahub.update(out)
            return True
        except Exception:
            return False

    def run(self):
        hdr_stage = 0  # 0:'A' 대기, 1:'B' 대기, 2:len/페이로드
        while True:
            try:
                if self.datahub.iscommunication_start:
                    if self.first_time:
                        self.setSerial(self.datahub.mySerialPort, self.datahub.myBaudrate)
                        self.first_time = False

                    if not self.ser.is_open:
                        self.ser.open()

                    self.datahub.serial_port_error = 0

                    b = self.ser.read(1)
                    if not b:
                        continue

                    if hdr_stage == 0:
                        hdr_stage = 1 if b == b'A' else 0

                    elif hdr_stage == 1:
                        hdr_stage = 2 if b == b'B' else 0

                    elif hdr_stage == 2:
                        # len 바이트
                        msg_len = b[0]
                        # len만큼 본문('Z' 포함) 읽기
                        payload = self.ser.read(msg_len)

                        if self._parse_and_update(payload):
                            # 정상 처리
                            pass
                        # 파싱 실패시에도 헤더 재동기화
                        hdr_stage = 0

                else:
                    # 통신 OFF → 포트 닫고 잠깐 쉼
                    if self.ser is not None and self.ser.is_open:
                        self.ser.close()
                    sleep(0.05)

            except Exception:
                # 포트 에러로 가정
                self.datahub.serial_port_error = 1
                hdr_stage = 0
                sleep(0.05)


if __name__ == "__main__":
    # 예시: datahub 주입 필요
    # receiver = Receiver(datahub)
    # receiver.start()
    pass
