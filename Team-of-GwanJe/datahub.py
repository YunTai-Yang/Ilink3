import numpy as np
import time
import threading

class Datahub:
    def __init__(self):
        # ---------- 통신 상태 ----------
        self.iscommunication_start = False
        self.isdatasaver_start = 0
        self.file_Name = 'Your File Name.csv'
        self.mySerialPort = 'COM8'
        self.mySendSerialPort = 'COM1'
        self.myBaudrate = 115200
        self.mySendBaudrate = 115200
        self.serial_port_error = -1
        self.sender_error = -1              # Sender용: -1(대기), 0(정상), 1(오류)

        # 스레드 락 (수신/시각화 경합 방지)
        self.lock = threading.Lock()

        # ---------- 시간 ----------
        self.hours    = np.empty(0, dtype=np.uint8)
        self.mins     = np.empty(0, dtype=np.uint8)
        self.secs     = np.empty(0, dtype=np.uint8)
        self.tenmilis = np.empty(0, dtype=np.uint8)
        self.t        = np.empty(0, dtype=np.float32)  # 누적초

        # ---------- ECEF 입력 (이름은 East/North/Up이지만 ECEF임: X/Y/Z) ----------
        self.Easts  = np.empty(0, dtype=np.float32)  # X
        self.Norths = np.empty(0, dtype=np.float32)  # Y
        self.Ups    = np.empty(0, dtype=np.float32)  # Z

        # ---------- ECEF 속도 입력 ----------
        self.vE = np.empty(0, dtype=np.float32)  # Xdot
        self.vN = np.empty(0, dtype=np.float32)  # Ydot
        self.vU = np.empty(0, dtype=np.float32)  # Zdot

        # ---------- ENU 출력 ----------
        self.e_enu  = np.empty(0, dtype=np.float32)
        self.n_enu  = np.empty(0, dtype=np.float32)
        self.u_enu  = np.empty(0, dtype=np.float32)
        self.vE_enu = np.empty(0, dtype=np.float32)
        self.vN_enu = np.empty(0, dtype=np.float32)
        self.vU_enu = np.empty(0, dtype=np.float32)

        # ---------- 스칼라 속도/호환 ----------
        self.speed  = np.empty(0, dtype=np.float32)
        self.yspeed = np.empty(0, dtype=np.float32)
        self.zspeed = np.empty(0, dtype=np.float32)

        # ---------- 센서/자세 ----------
        self.Xaccels = np.empty(0, dtype=np.float32)
        self.Yaccels = np.empty(0, dtype=np.float32)
        self.Zaccels = np.empty(0, dtype=np.float32)

        self.q0 = np.empty(0, dtype=np.float32)
        self.q1 = np.empty(0, dtype=np.float32)
        self.q2 = np.empty(0, dtype=np.float32)
        self.q3 = np.empty(0, dtype=np.float32)

        self.rollSpeeds  = np.empty(0, dtype=np.float32)
        self.pitchSpeeds = np.empty(0, dtype=np.float32)
        self.yawSpeeds   = np.empty(0, dtype=np.float32)

        self.rolls  = np.empty(0, dtype=np.float32)
        self.pitchs = np.empty(0, dtype=np.float32)
        self.yaws   = np.empty(0, dtype=np.float32)

        # ---------- 버튼 ----------
        self.button_data  = np.array([], dtype=np.uint8)
        self.button_names = [
            "launch", "launch_stop", "emergency_parachute",
            "staging_stop", "emergency_staging",  # ← 오타 수정
            "nc1_button", "nc2_button", "nc3_button"
        ]

        # ---------- ENU 기준 ----------
        self._ref_lla  = None  # (lat, lon, h)
        self._ref_ecef = None  # np.array([x,y,z])
        self.use_ref_for_euler = True  # True면 ref LLA 기준 ENU로 오일러 산출

        # map view trigger
        self.trigger_python = 0

    # ---------- 버튼 ----------
    def latest_button(self):
        if self.button_data.size == 0:
            return None
        return int(self.button_data[-1])

    def button_bit(self, idx):
        if self.button_data.size == 0 or not (0 <= idx <= 7):
            return None
        val = int(self.button_data[-1])
        return bool((val >> idx) & 0x01)

    # ---------- 좌표 변환 ----------
    @staticmethod
    def _lla_to_ecef(lat_deg, lon_deg, h_m):
        a = 6378137.0
        f = 1/298.257223563
        e2 = f*(2-f)
        lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
        sl, cl = np.sin(lat), np.cos(lat)
        so, co = np.sin(lon), np.cos(lon)
        N = a/np.sqrt(1 - e2*sl*sl)
        x = (N+h_m)*cl*co; y = (N+h_m)*cl*so; z = (N*(1-e2)+h_m)*sl
        return np.array([x, y, z], dtype=float)

    @staticmethod
    def _ecef_to_lla(x, y, z):
        a = 6378137.0
        f = 1/298.257223563
        e2 = f*(2-f)
        b = a*(1-f)
        ep2 = (a*a - b*b)/(b*b)
        r = np.hypot(x, y)
        if r < 1e-9:
            lat = np.sign(z) * np.pi/2; lon = 0.0; h = abs(z) - b
            return np.rad2deg(lat), np.rad2deg(lon), h
        lon = np.arctan2(y, x)
        theta = np.arctan2(z*a, r*b)
        st, ct = np.sin(theta), np.cos(theta)
        lat = np.arctan2(z + ep2*b*st**3, r - e2*a*ct**3)
        N = a/np.sqrt(1 - e2*np.sin(lat)**2)
        h = r/np.cos(lat) - N
        # Bowring 보정
        N = a/np.sqrt(1 - e2*np.sin(lat)**2)
        h = r/np.cos(lat) - N
        lat = np.arctan2(z, r*(1 - e2*N/(N+h)))
        return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(h)

    @staticmethod
    def _ecef_to_enu(xyz, ref_ecef, ref_lat_deg, ref_lon_deg):
        lat = np.deg2rad(ref_lat_deg); lon = np.deg2rad(ref_lon_deg)
        sl, cl = np.sin(lat), np.cos(lat)
        so, co = np.sin(lon), np.cos(lon)
        dx, dy, dz = xyz - ref_ecef
        T = np.array([[-so,        co,      0],
                      [-sl*co,  -sl*so,    cl],
                      [ cl*co,   cl*so,    sl]])
        return T @ np.array([dx, dy, dz], float)

    @staticmethod
    def _ecef_vec_to_enu(vec_xyz, ref_lat_deg, ref_lon_deg):
        lat = np.deg2rad(ref_lat_deg); lon = np.deg2rad(ref_lon_deg)
        sl, cl = np.sin(lat), np.cos(lat)
        so, co = np.sin(lon), np.cos(lon)
        T = np.array([[-so,        co,      0],
                      [-sl*co,  -sl*so,    cl],
                      [ cl*co,   cl*so,    sl]])
        return T @ np.asarray(vec_xyz, float)

    def set_reference_lla(self, lat_deg, lon_deg, h_m=0.0):
        self._ref_lla  = (float(lat_deg), float(lon_deg), float(h_m))
        self._ref_ecef = self._lla_to_ecef(*self._ref_lla)

    def _ensure_ref_from_first_ecef(self, x, y, z):
        if self._ref_ecef is None or self._ref_lla is None:
            lat0, lon0, h0 = self._ecef_to_lla(x, y, z)
            self._ref_lla  = (lat0, lon0, h0)
            self._ref_ecef = np.array([x, y, z], float)

    # ---------- 통신 제어 ----------
    def communication_start(self):
        self.iscommunication_start = True

    def communication_stop(self):
        self.iscommunication_start = False

    def datasaver_start(self):
        self.isdatasaver_start = 1

    def datasaver_stop(self):
        self.isdatasaver_start = 0

    def check_communication_error(self, timeout=2.0, poll_ms=50):
        """serial_port_error가 -1에서 0/1로 바뀔 때까지 기다림. 0=정상, 1=에러."""
        deadline = time.perf_counter() + timeout
        while True:
            s = self.serial_port_error
            if s in (0, 1):
                return s
            if time.perf_counter() >= deadline:
                return 1
            time.sleep(poll_ms / 1000.0)

    def check_sender_error(self, timeout=2.0, poll_ms=50):
        """
        sender_error가 -1에서 0/1로 바뀔 때까지 기다림. 0=정상, 1=에러.
        Sender 스레드가 전송 성공 시 0, 실패 시 1을 설정해야 함.
        """
        deadline = time.perf_counter() + timeout
        while True:
            s = self.sender_error
            if s in (0, 1):
                return s
            if time.perf_counter() >= deadline:
                return 1
            time.sleep(poll_ms / 1000.0)

    # ---------- 오일러 변환 유틸 ----------
    @staticmethod
    def _normalize_quat(w, x, y, z):
        q = np.array([w, x, y, z], float)
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n == 0.0:
            return 1.0, 0.0, 0.0, 0.0
        q /= n
        if q[0] < 0:
            q = -q
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])

    @staticmethod
    def quat_to_dcm_body_to_ecef(w, x, y, z):
        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        return np.array([
            [1-2*(yy+zz),   2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1-2*(xx+zz),   2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(xx+yy)]
        ], float)

    @staticmethod
    def euler_zxy_from_R(R):
        pitch_x = np.arcsin(-R[1, 2])
        cP = np.cos(pitch_x)
        EPS = 1e-6
        if abs(cP) < EPS:
            roll_z = 0.0
            sP = np.sign(-R[1, 2])
            yaw_y = np.arctan2(sP * R[0, 1], R[0, 0])
        else:
            roll_z = np.arctan2(R[1, 0], R[1, 1])
            yaw_y  = np.arctan2(R[0, 2], R[2, 2])
        return (np.degrees(roll_z), np.degrees(pitch_x), np.degrees(yaw_y))

    @staticmethod
    def ecef_to_enu_R(lat_deg, lon_deg):
        lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
        sl, cl = np.sin(lat), np.cos(lat)
        so, co = np.sin(lon), np.cos(lon)
        return np.array([[-so,        co,       0],
                         [-sl*co,  -sl*so,    cl],
                         [ cl*co,   cl*so,    sl]], float)

    @staticmethod
    def euler_from_quat_body_to_ecef_zxy(w, x, y, z):
        w, x, y, z = Datahub._normalize_quat(w, x, y, z)
        R_be = Datahub.quat_to_dcm_body_to_ecef(w, x, y, z)
        return Datahub.euler_zxy_from_R(R_be)

    @staticmethod
    def euler_from_quat_body_to_enu_zxy(w, x, y, z, lat_deg, lon_deg):
        w, x, y, z = Datahub._normalize_quat(w, x, y, z)
        R_be = Datahub.quat_to_dcm_body_to_ecef(w, x, y, z)   # body -> ECEF
        R_en = Datahub.ecef_to_enu_R(lat_deg, lon_deg)        # ECEF -> ENU
        R_bn = R_en @ R_be                                    # body -> ENU
        return Datahub.euler_zxy_from_R(R_bn)

    # ---------- 실시간 패킷 갱신 ----------
    def update(self, datas):
        """
        datas: 길이 20
        [h,m,s,tm, E,N,U, vE,vN,vU, a_p,a_y,a_r, q0,q1,q2,q3, w_p,w_y,w_r]
        """
        if datas is None or len(datas) < 20:
            return

        h = int(datas[0]); m = int(datas[1]); s = int(datas[2]); tm = int(datas[3])
        E, N, U       = map(float, datas[4:7])
        vE, vN, vU    = map(float, datas[7:10])
        a_p, a_y, a_r = map(float, datas[10:13])
        q0, q1, q2, q3 = map(float, datas[13:17])
        w_p, w_y, w_r  = map(float, datas[17:20])

        # 누적 시간(sec)
        tsec = float(h)*3600.0 + float(m)*60.0 + float(s) + float(tm)/100.0

        with self.lock:
            # time
            self.hours    = np.append(self.hours,    np.uint8(h))
            self.mins     = np.append(self.mins,     np.uint8(m))
            self.secs     = np.append(self.secs,     np.uint8(s))
            self.tenmilis = np.append(self.tenmilis, np.uint8(tm))
            self.t        = np.append(self.t,        np.float32(tsec))

            # ECEF pos/vel
            self.Easts  = np.append(self.Easts,  np.float32(E))
            self.Norths = np.append(self.Norths, np.float32(N))
            self.Ups    = np.append(self.Ups,    np.float32(U))

            self.vE = np.append(self.vE, np.float32(vE))
            self.vN = np.append(self.vN, np.float32(vN))
            self.vU = np.append(self.vU, np.float32(vU))

            # accel / gyro
            self.Xaccels = np.append(self.Xaccels, np.float32(a_p))
            self.Yaccels = np.append(self.Yaccels, np.float32(a_y))
            self.Zaccels = np.append(self.Zaccels, np.float32(a_r))

            self.q0 = np.append(self.q0, np.float32(q0))
            self.q1 = np.append(self.q1, np.float32(q1))
            self.q2 = np.append(self.q2, np.float32(q2))
            self.q3 = np.append(self.q3, np.float32(q3))

            self.rollSpeeds  = np.append(self.rollSpeeds,  np.float32(w_p))
            self.pitchSpeeds = np.append(self.pitchSpeeds, np.float32(w_y))
            self.yawSpeeds   = np.append(self.yawSpeeds,   np.float32(w_r))

            # ENU 기준 확보
            self._ensure_ref_from_first_ecef(E, N, U)
            lat0, lon0, _ = self._ref_lla

            # Euler (body→ENU, Z–X–Y)
            if self.use_ref_for_euler:
                r, p, y_ = Datahub.euler_from_quat_body_to_enu_zxy(q0, q1, q2, q3, lat0, lon0)
            else:
                lat_deg, lon_deg, _ = self._ecef_to_lla(E, N, U)
                r, p, y_ = Datahub.euler_from_quat_body_to_enu_zxy(q0, q1, q2, q3, lat_deg, lon_deg)

            self.rolls  = np.append(self.rolls,  np.float32(r))
            self.pitchs = np.append(self.pitchs, np.float32(p))
            self.yaws   = np.append(self.yaws,   np.float32(y_))

            # ENU pos/vel
            e, n, u = self._ecef_to_enu(np.array([E, N, U], float), self._ref_ecef, lat0, lon0)
            self.e_enu = np.append(self.e_enu, np.float32(e))
            self.n_enu = np.append(self.n_enu, np.float32(n))
            self.u_enu = np.append(self.u_enu, np.float32(u))

            vE_e, vN_e, vU_e = self._ecef_vec_to_enu(np.array([vE, vN, vU], float), lat0, lon0)
            self.vE_enu = np.append(self.vE_enu, np.float32(vE_e))
            self.vN_enu = np.append(self.vN_enu, np.float32(vN_e))
            self.vU_enu = np.append(self.vU_enu, np.float32(vU_e))

            spd = float(np.sqrt(vE_e*vE_e + vN_e*vN_e + vU_e*vU_e))
            self.speed  = np.append(self.speed,  np.float32(spd))
            self.yspeed = np.append(self.yspeed, np.float32(vN_e))
            self.zspeed = np.append(self.zspeed, np.float32(vU_e))

    # ---------- CSV/행 입력 ----------
    def update_from_row(self, row):
        (hours, mins, secs, tenmilis,
         x, y, z, vx, vy, vz,
         a_p, a_y, a_r,
         q0, q1, q2, q3,
         w_p, w_y, w_r, *rest) = row

        with self.lock:
            self.hours    = np.append(self.hours,    np.uint8(hours))
            self.mins     = np.append(self.mins,     np.uint8(mins))
            self.secs     = np.append(self.secs,     np.uint8(secs))
            self.tenmilis = np.append(self.tenmilis, np.uint8(tenmilis))

            self.Easts  = np.append(self.Easts,  np.float32(x))
            self.Norths = np.append(self.Norths, np.float32(y))
            self.Ups    = np.append(self.Ups,    np.float32(z))

            self.vE = np.append(self.vE, np.float32(vx))
            self.vN = np.append(self.vN, np.float32(vy))
            self.vU = np.append(self.vU, np.float32(vz))

            self.Xaccels     = np.append(self.Xaccels,     np.float32(a_p))
            self.Yaccels     = np.append(self.Yaccels,     np.float32(a_y))
            self.Zaccels     = np.append(self.Zaccels,     np.float32(a_r))
            self.rollSpeeds  = np.append(self.rollSpeeds,  np.float32(w_p))
            self.pitchSpeeds = np.append(self.pitchSpeeds, np.float32(w_y))
            self.yawSpeeds   = np.append(self.yawSpeeds,   np.float32(w_r))

            self.q0 = np.append(self.q0, np.float32(q0))
            self.q1 = np.append(self.q1, np.float32(q1))
            self.q2 = np.append(self.q2, np.float32(q2))
            self.q3 = np.append(self.q3, np.float32(q3))

            self._ensure_ref_from_first_ecef(float(x), float(y), float(z))
            lat0, lon0, _ = self._ref_lla

            r, p, y_ = Datahub.euler_from_quat_body_to_enu_zxy(q0, q1, q2, q3, lat0, lon0)
            self.rolls  = np.append(self.rolls,  np.float32(r))
            self.pitchs = np.append(self.pitchs, np.float32(p))
            self.yaws   = np.append(self.yaws,   np.float32(y_))

            e, n, u = self._ecef_to_enu(np.array([x, y, z], float), self._ref_ecef, lat0, lon0)
            self.e_enu = np.append(self.e_enu, np.float32(e))
            self.n_enu = np.append(self.n_enu, np.float32(n))
            self.u_enu = np.append(self.u_enu, np.float32(u))

            vE_e, vN_e, vU_e = self._ecef_vec_to_enu(np.array([vx, vy, vz], float), lat0, lon0)
            self.vE_enu = np.append(self.vE_enu, np.float32(vE_e))
            self.vN_enu = np.append(self.vN_enu, np.float32(vN_e))
            self.vU_enu = np.append(self.vU_enu, np.float32(vU_e))

    def clear(self):
        self.__init__()