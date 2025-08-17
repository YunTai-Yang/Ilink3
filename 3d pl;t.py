import threading
import time
from collections import deque
from queue import Queue, Empty

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class RealTimeENUPlot:
    def __init__(self,
                 tail_len: int = 200,        # 그릴 최대 포인트 수(궤적 길이)
                 update_hz: float = 20.0,    # 화면 업데이트 주기(Hz)
                 vel_scale: float = 1.0,     # 속도 벡터 길이 스케일
                 auto_scale: bool = True,    # 축 오토스케일 여부
                 margin: float = 5.0):       # 오토스케일 여백
        self.tail_len = tail_len
        self.dt = 1.0 / update_hz
        self.vel_scale = vel_scale
        self.auto_scale = auto_scale
        self.margin = margin

        # 실시간 데이터 저장
        self.E = deque(maxlen=tail_len)
        self.N = deque(maxlen=tail_len)
        self.U = deque(maxlen=tail_len)
        self.vE = 0.0
        self.vN = 0.0
        self.vU = 0.0

        # 외부 입력용 큐
        self.queue = Queue()

        # Figure / Axes / Artists
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("E (m)")
        self.ax.set_ylabel("N (m)")
        self.ax.set_zlabel("U (m)")
        self.ax.set_title("Rocket ENU & Velocity (Real-time)")

        # 라인(궤적), 최신 포인트
        (self.traj_line,) = self.ax.plot([], [], [], lw=1)   # 색상 지정 안 함(기본값)
        self.head_scatter = self.ax.scatter([], [], [], s=25)

        # 3D quiver(속도 벡터). 매 프레임 재생성(간단/안전)
        self.vel_quiver = None

        # 애니메이션
        self.ani = FuncAnimation(
            self.fig,
            self._update_frame,
            init_func=self._init_anim,
            interval=int(self.dt * 1000),
            blit=False  # 3D에선 blit 비권장
        )

    # 외부에서 데이터를 넣는 진입점
    def push_sample(self, E: float, N: float, U: float, vE: float, vN: float, vU: float):
        self.queue.put((E, N, U, vE, vN, vU))

    def _drain_queue(self):
        """큐에 쌓인 시점까지 최신 상태로 반영(버퍼가 빠르게 들어와도 프레임당 한 번에 처리)."""
        updated = False
        while True:
            try:
                E, N, U, vE, vN, vU = self.queue.get_nowait()
                self.E.append(E)
                self.N.append(N)
                self.U.append(U)
                self.vE, self.vN, self.vU = vE, vN, vU
                updated = True
            except Empty:
                break
        return updated

    def _init_anim(self):
        self.traj_line.set_data([], [])
        self.traj_line.set_3d_properties([])
        # scatter 초기화
        self.head_scatter._offsets3d = ([], [], [])
        return self.traj_line, self.head_scatter

    def _autoscale_axes(self):
        if not self.E:
            return
        e = np.array(self.E)
        n = np.array(self.N)
        u = np.array(self.U)
        e_min, e_max = e.min() - self.margin, e.max() + self.margin
        n_min, n_max = n.min() - self.margin, n.max() + self.margin
        u_min, u_max = u.min() - self.margin, u.max() + self.margin
        self.ax.set_xlim(e_min, e_max)
        self.ax.set_ylim(n_min, n_max)
        self.ax.set_zlim(u_min, u_max)

        # 축 비율 동일(등축)로 맞추기 (3D에서 왜곡 방지)
        max_range = max(e_max - e_min, n_max - n_min, u_max - u_min)
        e_mid = (e_max + e_min) / 2.0
        n_mid = (n_max + n_min) / 2.0
        u_mid = (u_max + u_min) / 2.0
        r = max_range / 2.0
        self.ax.set_xlim(e_mid - r, e_mid + r)
        self.ax.set_ylim(n_mid - r, n_mid + r)
        self.ax.set_zlim(u_mid - r, u_mid + r)

    def _update_frame(self, _):
        # 최신 큐 비우고 모두 반영
        updated = self._drain_queue()
        if not updated and not self.E:
            return self.traj_line, self.head_scatter

        # 라인/헤드 갱신
        self.traj_line.set_data(self.E, self.N)
        self.traj_line.set_3d_properties(self.U)

        e_last = self.E[-1]
        n_last = self.N[-1]
        u_last = self.U[-1]
        self.head_scatter._offsets3d = ([e_last], [n_last], [u_last])

        # 속도 벡터(기존 quiver 제거 후 재생성)
        if self.vel_quiver is not None:
            self.vel_quiver.remove()
            self.vel_quiver = None

        # 벡터의 시작점: 최신 위치, 성분: vE, vN, vU
        self.vel_quiver = self.ax.quiver(
            e_last, n_last, u_last,
            self.vE * self.vel_scale, self.vN * self.vel_scale, self.vU * self.vel_scale,
            length=1.0, normalize=False
        )

        if self.auto_scale:
            self._autoscale_axes()

        return self.traj_line, self.head_scatter, self.vel_quiver

    def show(self):
        plt.show()


# -----------------------------
# 사용 예시(데이터 스트림 시뮬레이터)
# -----------------------------
def _sim_data_source(plot: RealTimeENUPlot, hz: float = 50.0):
    """실제 GPS/INS 파이프라인 대신 임의 궤적 + 속도 생성(데모용)."""
    dt = 1.0 / hz
    t = 0.0
    e, n, u = 0.0, 0.0, 0.0
    vE, vN, vU = 0.0, 0.0, 0.0
    while True:
        # 간단한 나선형 궤적 + 약간의 잡음
        vE = 10.0 * np.cos(0.2 * t)
        vN = 10.0 * np.sin(0.2 * t)
        vU = 1.0 * np.sin(0.05 * t)

        e += vE * dt
        n += vN * dt
        u += vU * dt

        plot.push_sample(e, n, u, vE, vN, vU)
        time.sleep(dt)
        t += dt


if __name__ == "__main__":
    # 플로터 생성(필요 시 파라미터 조정)
    rtp = RealTimeENUPlot(
        tail_len=400,
        update_hz=20.0,
        vel_scale=0.2,   # 속도 벡터 길이 조정
        auto_scale=True
    )

    # 실데이터가 있다면, 아래 스레드 대신 센서/통신 콜백에서 rtp.push_sample(...) 호출
    th = threading.Thread(target=_sim_data_source, args=(rtp,), daemon=True)
    th.start()

    rtp.show()
