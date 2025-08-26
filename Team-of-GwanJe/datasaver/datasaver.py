from csv import writer
from time import sleep
import numpy as np
import os

class DataSaver:
    def __init__(self, datahub):
        self.datahub = datahub
        self.file = None
        self.writer = None
        self.saverows = 0
        self.log_dir = 'log'  # 저장 폴더

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    # ------- 내부 유틸 -------
    def _snapshot_fields(self):
        """락을 잡고 모든 컬럼을 같은 순서로 스냅샷(복사본) 반환."""
        dh = self.datahub
        with dh.lock:
            fields = [
                dh.hours.copy(),          # h
                dh.mins.copy(),           # m
                dh.secs.copy(),           # s
                dh.tenmilis.copy(),       # ms (10ms 단위)
                dh.Easts.copy(),          # ecef_x (X)
                dh.Norths.copy(),         # ecef_y (Y)
                dh.Ups.copy(),            # ecef_z (Z)
                dh.vE.copy(),             # v_ecef_x (Xd)
                dh.vN.copy(),             # v_ecef_y (Yd)
                dh.vU.copy(),             # v_ecef_z (Zd)
                dh.Xaccels.copy(),        # a_p
                dh.Yaccels.copy(),        # a_y
                dh.Zaccels.copy(),        # a_r
                dh.q0.copy(),             # q0
                dh.q1.copy(),             # q1
                dh.q2.copy(),             # q2
                dh.q3.copy(),             # q3
                dh.rollSpeeds.copy(),     # w_p
                dh.pitchSpeeds.copy(),    # w_y
                dh.yawSpeeds.copy(),      # w_r
            ]
        return fields

    @staticmethod
    def _min_len(arrs):
        if not arrs:
            return 0
        try:
            return min(len(a) for a in arrs)
        except Exception:
            # None 등 섞여있을 경우 대비
            lengths = [(len(a) if a is not None else 0) for a in arrs]
            return min(lengths) if lengths else 0

    def _write_new_rows(self):
        """스냅샷 기준으로 saverows 이후의 신규 행만 CSV에 추가."""
        fields = self._snapshot_fields()
        ready_len = self._min_len(fields)
        line_remain = ready_len - self.saverows
        if line_remain <= 0:
            return 0

        # 각 컬럼을 동일 범위로 슬라이싱
        start = self.saverows
        end = start + line_remain
        sliced = [col[start:end] for col in fields]

        # (행,열)로 만들기: column-major -> row-major
        # numpy.column_stack은 길이 동일한 1D 배열 리스트를 2D (N, M)로 쌓음
        mat = np.column_stack(sliced)  # shape: (line_remain, 20)

        # csv.writer는 1D iterable을 한 행으로 씀
        for row in mat:
            # numpy 타입 그대로 넣어도 되지만, 안전하게 파이썬 기본형으로 캐스팅
            self.writer.writerow([*map(lambda x: x.item() if hasattr(x, 'item') else x, row)])

        self.file.flush()
        self.saverows += line_remain
        return line_remain

    # ------- 메인 루프 -------
    def start(self):
        """datasaver 플래그를 감시하며 파일 열고/닫고/추가 저장."""
        while True:
            if self.datahub.isdatasaver_start:
                # 파일 오픈 및 헤더 기록 (덮어쓰기)
                if self.file is None or self.file.closed:
                    self.saverows = 0
                    path = os.path.join(self.log_dir, self.datahub.file_Name)
                    self.file = open(path, 'w', newline='')
                    self.writer = writer(self.file)
                    self.writer.writerow([
                        "h", "m", "s", "ms",
                        "ecef_x", "ecef_y", "ecef_z",
                        "v_ecef_x", "v_ecef_y", "v_ecef_z",
                        "a_p", "a_y", "a_r",
                        "q0", "q1", "q2", "q3",
                        "w_p", "w_y", "w_r"
                    ])

                # 저장 루프 (플래그가 내려가면 빠져나가 닫기)
                while self.datahub.isdatasaver_start:
                    wrote = self._write_new_rows()
                    sleep(0.02 if wrote else 0.05)

                # 플래그 내려감 → 파일 닫기
                if self.file is not None and not self.file.closed:
                    self.file.close()
                self.saverows = 0

            # 대기 (datasaver 시작 대기)
            sleep(0.1)

    def stop(self):
        if self.file is not None and not self.file.closed:
            self.file.close()
        self.saverows = 0