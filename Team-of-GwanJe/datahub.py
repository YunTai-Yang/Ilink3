from numpy import empty, append
import numpy as np

class Datahub:
    def __init__(self):
        """
        Communication Status Parameter
        """
        self.iscommunication_start = False
        self.isdatasaver_start = 0
        self.file_Name = 'Your File Name.csv'
        self.mySerialPort = 'COM8'
        self.myBaudrate = 115200
        self.serial_port_error = -1

        """
        Rocket Status Parameter
        """
        self.hours       = empty(0)
        self.mins        = empty(0)
        self.secs        = empty(0)
        self.tenmilis    = empty(0)
        self.t           = empty(0)

        # ECEF position (입력)
        self.Easts       = empty(0)   # X
        self.Norths      = empty(0)   # Y
        self.Ups         = empty(0)   # Z

        # ECEF velocity (입력)
        self.vE          = empty(0)   # Xdot
        self.vN          = empty(0)   # Ydot
        self.vU          = empty(0)   # Zdot

        # ENU position/velocity (출력 - 신규)
        self.e_enu       = empty(0)   # East
        self.n_enu       = empty(0)   # North
        self.u_enu       = empty(0)   # Up
        self.vE_enu      = empty(0)
        self.vN_enu      = empty(0)
        self.vU_enu      = empty(0)

        # 기타
        self.speed       = empty(0)
        self.yspeed      = empty(0)
        self.zspeed      = empty(0)

        self.Xaccels     = empty(0)
        self.Yaccels     = empty(0)
        self.Zaccels     = empty(0)

        self.q0          = empty(0)
        self.q1          = empty(0)
        self.q2          = empty(0)
        self.q3          = empty(0)

        self.rollSpeeds  = empty(0)
        self.pitchSpeeds = empty(0)
        self.yawSpeeds   = empty(0)

        self.rolls       = empty(0)
        self.pitchs      = empty(0)
        self.yaws        = empty(0)

        self.button_data = np.array([], dtype=np.uint8)
        self.button_names = ["launch","launch_stop","emergency_parachute","staging_stop","mergency_staging","nc1_button","nc2_button","nc3_button"]


        # ENU 기준 (자동/수동 설정)
        self._ref_lla  = None   # (lat, lon, h)
        self._ref_ecef = None   # np.array([x,y,z])

        # map view trigger
        self.trigger_python = 0

    def latest_button(self):
        """최근 버튼 데이터 1바이트 반환 (없으면 None)"""
        if self.button_data.size == 0:
            return None
        return int(self.button_data[-1])

    def button_bit(self, idx):
        """특정 비트 확인 (idx=0~7). True/False 반환"""
        if self.button_data.size == 0:
            return None
        if not (0 <= idx <= 7):
            return None
        val = int(self.button_data[-1])
        return bool((val >> idx) & 0x01)

    #uint8 안전 변환
    @staticmethod
    def _to_u8(val):
        try:
            i = int(val)
        except Exception:
            return np.uint8(0)
        return np.uint8(i & 0xFF)

    # -------- 좌표변환 유틸 --------
    @staticmethod
    def _lla_to_ecef(lat_deg, lon_deg, h_m):
        a = 6378137.0; f = 1/298.257223563; e2 = f*(2-f)
        lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
        sl, cl = np.sin(lat), np.cos(lat)
        so, co = np.sin(lon), np.cos(lon)
        N = a/np.sqrt(1 - e2*sl*sl)
        x = (N+h_m)*cl*co; y = (N+h_m)*cl*so; z = (N*(1-e2)+h_m)*sl
        return np.array([x, y, z], float)

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
        # Bowring 1-step
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
        # 속도/가속도 같은 '벡터'는 회전만 적용 (평행이동 없음)
        lat = np.deg2rad(ref_lat_deg); lon = np.deg2rad(ref_lon_deg)
        sl, cl = np.sin(lat), np.cos(lat)
        so, co = np.sin(lon), np.cos(lon)
        T = np.array([[-so,        co,      0],
                      [-sl*co,  -sl*so,    cl],
                      [ cl*co,   cl*so,    sl]])
        return T @ np.asarray(vec_xyz, float)

    def set_reference_lla(self, lat_deg, lon_deg, h_m=0.0):
        """외부에서 기준 ENU 원점을 명시적으로 지정하고 싶을 때 사용"""
        self._ref_lla  = (float(lat_deg), float(lon_deg), float(h_m))
        self._ref_ecef = self._lla_to_ecef(*self._ref_lla)

    def _ensure_ref_from_first_ecef(self, x, y, z):
        if self._ref_ecef is None or self._ref_lla is None:
            lat0, lon0, h0 = self._ecef_to_lla(x, y, z)
            self._ref_lla  = (lat0, lon0, h0)
            self._ref_ecef = np.array([x, y, z], float)

    # -------- 통신/상태 --------
    def communication_start(self):
        self.iscommunication_start = True

    def communication_stop(self):
        self.iscommunication_start = False

    def check_communication_error(self):
        while True:
            if self.serial_port_error in (0, 1):
                return self.serial_port_error

    def datasaver_start(self):
        self.isdatasaver_start = 1

    def datasaver_stop(self):
        self.isdatasaver_start = 0

    # --- Z–X–Y Euler helpers (roll about Z, pitch about X, yaw about Y) ---
    @staticmethod
    def _normalize_quat(w, x, y, z):
        q = np.array([w, x, y, z], float)
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n == 0.0:
            return 1.0, 0.0, 0.0, 0.0
        q /= n
        # 연속성 위해 w<0이면 부호 뒤집기 (옵션)
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
        """
        Extrinsic Z–X–Y 분해.
        roll_z  = rotation about Z
        pitch_x = rotation about X
        yaw_y   = rotation about Y
        반환: (roll_z, pitch_x, yaw_y) [deg]
        R = Ry(yaw_y) * Rx(pitch_x) * Rz(roll_z)
        """
        # 기본식
        pitch_x = np.arcsin(-R[1, 2])                  # R[1,2] = -sin(pitch)
        cP = np.cos(pitch_x)
        EPS = 1e-6

        # 짐벌락 보호
        if abs(cP) < EPS:
            # pitch_x ≈ ±90° → roll_z와 yaw_y가 커플링
            roll_z = 0.0
            sP = np.sign(-R[1,2])  # ≈ ±1
            yaw_y  = np.arctan2(sP * R[0,1], R[0,0])
        else:
            roll_z = np.arctan2(R[1,0], R[1,1])        # tan(roll) = R10 / R11  (cP 약분)
            yaw_y  = np.arctan2(R[0,2], R[2,2])        # tan(yaw)  = R02 / R22  (cP 약분)

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
        """바디→ECEF quat → Z–X–Y (roll_z, pitch_x, yaw_y)[deg]"""
        w, x, y, z = Datahub._normalize_quat(w, x, y, z)
        R_be = Datahub.quat_to_dcm_body_to_ecef(w, x, y, z)
        return Datahub.euler_zxy_from_R(R_be)

    @staticmethod
    def euler_from_quat_body_to_enu_zxy(w, x, y, z, lat_deg, lon_deg):
        """바디→ECEF quat + (lat,lon) → ENU 기준 Z–X–Y 오일러[deg]"""
        w, x, y, z = Datahub._normalize_quat(w, x, y, z)
        R_be = Datahub.quat_to_dcm_body_to_ecef(w, x, y, z)   # body -> ECEF
        R_en = Datahub.ecef_to_enu_R(lat_deg, lon_deg)        # ECEF -> ENU
        R_bn = R_en @ R_be                                     # body -> ENU
        return Datahub.euler_zxy_from_R(R_bn)

    # -------- 데이터 갱신 (실시간 패킷) --------
    def update(self, datas):
        """Update Datas received from rocket (ECEF pos/vel이 들어온다고 가정)"""
        # time
        self.hours       = append(self.hours,    datas[0])
        self.mins        = append(self.mins,     datas[1])
        self.secs        = append(self.secs,     datas[2])
        self.tenmilis    = append(self.tenmilis, datas[3])

        # ECEF pos/vel
        x = float(datas[4]); y = float(datas[5]); z = float(datas[6])
        vx = float(datas[7]); vy = float(datas[8]); vz = float(datas[9])

        self.Easts  = append(self.Easts,  x)
        self.Norths = append(self.Norths, y)
        self.Ups    = append(self.Ups,    z)

        self.vE = append(self.vE, vx)
        self.vN = append(self.vN, vy)
        self.vU = append(self.vU, vz)

        # 기타
        self.speed   = append(self.speed,   datas[10])
        self.yspeed  = append(self.yspeed,  datas[11])
        self.zspeed  = append(self.zspeed,  datas[12])

        self.Xaccels = append(self.Xaccels, datas[13])
        self.Yaccels = append(self.Yaccels, datas[14])
        self.Zaccels = append(self.Zaccels, datas[15])

        q0 = float(datas[16]); q1 = float(datas[17]); q2 = float(datas[18]); q3 = float(datas[19])
        self.q0 = append(self.q0, q0); self.q1 = append(self.q1, q1)
        self.q2 = append(self.q2, q2); self.q3 = append(self.q3, q3)

        self.rollSpeeds  = append(self.rollSpeeds,  datas[20])
        self.pitchSpeeds = append(self.pitchSpeeds, datas[21])
        self.yawSpeeds   = append(self.yawSpeeds,   datas[22])

        # --- Euler (권장: ENU 기준) ---
        # 현재 위치에서 lat/lon 얻기
        lat_deg, lon_deg, _ = self._ecef_to_lla(x, y, z)

        r, p, y_ = Datahub.euler_from_quat_body_to_enu_zxy(q0, q1, q2, q3, lat_deg, lon_deg)
        self.rolls  = append(self.rolls,  r)
        self.pitchs = append(self.pitchs, p)
        self.yaws   = append(self.yaws,   y_)

        # --- ECEF → ENU 변환 및 저장 (핵심 추가) ---
        self._ensure_ref_from_first_ecef(x, y, z)
        lat0, lon0, _ = self._ref_lla

        e, n, u = self._ecef_to_enu(np.array([x, y, z], float), self._ref_ecef, lat0, lon0)
        self.e_enu = append(self.e_enu, e)
        self.n_enu = append(self.n_enu, n)
        self.u_enu = append(self.u_enu, u)

        vE_e, vN_e, vU_e = self._ecef_vec_to_enu(np.array([vx, vy, vz], float), lat0, lon0)
        self.vE_enu = append(self.vE_enu, vE_e)
        self.vN_enu = append(self.vN_enu, vN_e)
        self.vU_enu = append(self.vU_enu, vU_e)

        # -------- 데이터 갱신 (CSV/row 등) --------
    def update_from_row(self, row):
        (hours, mins, secs, tenmilis,
        x, y, z, vx, vy, vz,
        a_p, a_y, a_r,
        q0, q1, q2, q3,
        w_p, w_y, w_r,
        *rest) = row   # row를 그대로 두고, 끝부분을 rest로 남겨둠

        # --- time ---
        self.hours    = append(self.hours,    hours)
        self.mins     = append(self.mins,     mins)
        self.secs     = append(self.secs,     secs)
        self.tenmilis = append(self.tenmilis, tenmilis)

        # --- ECEF pos/vel ---
        self.Easts  = append(self.Easts,  x)
        self.Norths = append(self.Norths, y)
        self.Ups    = append(self.Ups,    z)

        self.vE = append(self.vE, vx)
        self.vN = append(self.vN, vy)
        self.vU = append(self.vU, vz)

        # --- accel / gyro ---
        self.Xaccels     = append(self.Xaccels,     a_p)
        self.Yaccels     = append(self.Yaccels,     a_y)
        self.Zaccels     = append(self.Zaccels,     a_r)
        self.rollSpeeds  = append(self.rollSpeeds,  w_p)
        self.pitchSpeeds = append(self.pitchSpeeds, w_y)
        self.yawSpeeds   = append(self.yawSpeeds,   w_r)
    
        # --- quat & euler ---
        self.q0 = append(self.q0, q0)
        self.q1 = append(self.q1, q1)
        self.q2 = append(self.q2, q2)
        self.q3 = append(self.q3, q3)

        lat_deg, lon_deg, _ = self._ecef_to_lla(x, y, z)
        r, p, y_ = Datahub.euler_from_quat_body_to_enu_zxy(q0, q1, q2, q3, lat_deg, lon_deg)
        self.rolls  = append(self.rolls,  r)
        self.pitchs = append(self.pitchs, p)
        self.yaws   = append(self.yaws,   y_)

        # --- ECEF → ENU 변환 및 저장 ---
        self._ensure_ref_from_first_ecef(float(x), float(y), float(z))
        lat0, lon0, _ = self._ref_lla

        e, n, u = self._ecef_to_enu(np.array([x, y, z], float), self._ref_ecef, lat0, lon0)
        self.e_enu = append(self.e_enu, e)
        self.n_enu = append(self.n_enu, n)
        self.u_enu = append(self.u_enu, u)

        vE_e, vN_e, vU_e = self._ecef_vec_to_enu(np.array([vx, vy, vz], float), lat0, lon0)
        self.vE_enu = append(self.vE_enu, vE_e)
        self.vN_enu = append(self.vN_enu, vN_e)
        self.vU_enu = append(self.vU_enu, vU_e)

    def clear(self):
        self.__init__()