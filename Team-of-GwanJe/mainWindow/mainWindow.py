from PyQt5.QtWidgets import QShortcut, QApplication, QMainWindow, QGraphicsDropShadowEffect , QPushButton,QLineEdit, QLabel, QMessageBox, QInputDialog, QCheckBox, QStackedWidget, QAction, QFrame, QWidget, QVBoxLayout, QFileDialog
from pathlib import Path
from PyQt5.QtCore import QThread, QUrl, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QIcon, QPixmap, QVector3D, QKeySequence
import pyqtgraph as pg
from pyqtgraph import PlotWidget, GridItem, AxisItem
#################
import numpy as np
from numpy import array, deg2rad, cos, sin, cross
from numpy.linalg import norm
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QLabel
import pyqtgraph.opengl as gl
import sys
import os
import pandas as pd
import time
#################
from numpy import zeros, array, cross, reshape, sin, cos, deg2rad, rad2deg
from numpy.random import rand
from numpy.linalg import norm

from matplotlib.pyplot import figure
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt

from datetime import timedelta
from os.path import abspath, dirname, join, exists
from sys import exit, argv

from datahub import Datahub

from pandas import read_csv

from . import widgetStyle as ws

COLUMN_NAMES = [
    "hours","mins","secs","tenmilis","E","N","U","v_E","v_N","v_U","a_p","a_y","a_r","q_0","q_1","q_2","q_3","w_p","w_y","w_r"
]

class PageWindow(QMainWindow):
    gotoSignal = pyqtSignal(str)
    def goto(self, name):
        self.gotoSignal.emit(name)

class GraphViewer_Thread(QThread):
    def __init__(self, mainwindow,datahub):
        super().__init__()
        self.mainwindow = mainwindow
        self.datahub = datahub

        self.view = QWebEngineView(self.mainwindow.container)
        self.view.load(QUrl())
        self.view.setGeometry(*ws.webEngine_geometry)
        
        self.angleSpeed_title = QLabel(self.mainwindow.container)
        self.angleSpeed_title.setText("<b>&#8226; Angle Speed</b>")
        self.angleSpeed_title.setStyleSheet("color: white;")
        self.pw_angleSpeed = PlotWidget(self.mainwindow.container)
        
        self.accel_title = QLabel(self.mainwindow.container)
        self.accel_title.setText("<b>&#8226; Acceleration</b>")
        self.accel_title.setStyleSheet("color: white;")
        self.pw_accel = PlotWidget(self.mainwindow.container)
        
        self.speed_title = QLabel(self.mainwindow.container)
        self.speed_title.setText("<b>&#8226; Speed</b>")
        self.speed_title.setStyleSheet("color: white;")
        self.pw_speed = PlotWidget(self.mainwindow.container)
        
        self.pw_angleSpeed.setGeometry(*ws.pw_angleSpeed_geometry)
        self.pw_accel.setGeometry(*ws.pw_accel_geometry)
        self.pw_speed.setGeometry(*ws.pw_speed_geometry)
          
        self.angleSpeed_title.setGeometry(*ws.angleSpeed_title_geometry)
        self.accel_title.setGeometry(*ws.accel_title_geometry)
        self.speed_title.setGeometry(*ws.speed_title_geometry)

        self.angleSpeed_title.setFont(ws.font_angleSpeed_title)
        self.accel_title.setFont(ws.font_accel_title)
        self.speed_title.setFont(ws.font_speed_title)
        

        self.pw_angleSpeed.addItem(GridItem())
        self.pw_accel.addItem(GridItem())
        self.pw_speed.addItem(GridItem())

        #set label in each axis
        self.pw_angleSpeed.getPlotItem().getAxis('bottom').setLabel('Time(second)')
        self.pw_angleSpeed.getPlotItem().getAxis('left').setLabel('Degree/second')
        self.pw_accel.getPlotItem().getAxis('bottom').setLabel('Time(second)')
        self.pw_accel.getPlotItem().getAxis('left').setLabel('g(gravity accel)')
        self.pw_speed.getPlotItem().getAxis('bottom').setLabel('Time(second)')
        self.pw_speed.getPlotItem().getAxis('left').setLabel('speed(m/s)')
        #set range in each axis
        self.pw_angleSpeed.setYRange(-1000,1000)
        self.pw_accel.setYRange(-20,20)
        self.pw_speed.setYRange(-100,1000)

        #legend
        self.pw_angleSpeed.getPlotItem().addLegend()
        self.pw_accel.getPlotItem().addLegend()
        self.pw_speed.getPlotItem().addLegend()

        self.curve_rollSpeed = self.pw_angleSpeed.plot(pen='r', name = "roll speed")
        self.curve_pitchSpeed = self.pw_angleSpeed.plot(pen='g', name = "pitch speed")
        self.curve_yawSpeed = self.pw_angleSpeed.plot(pen='b', name = "yaw speed")

        self.curve_xaccel = self.pw_accel.plot(pen='r', name = "x acc")
        self.curve_yaccel = self.pw_accel.plot(pen='g', name = "y acc")
        self.curve_zaccel = self.pw_accel.plot(pen='b', name ="z acc")

        self.curve_xspeed = self.pw_speed.plot(pen='r', name = 'x speed')
        self.curve_yspeed = self.pw_speed.plot(pen='g', name = 'y speed')
        self.curve_zspeed = self.pw_speed.plot(pen='b', name = 'z speed')


        self.x_ran = 500
        self.time = zeros(self.x_ran)
        self.rollSpeed = zeros(self.x_ran)
        self.pitchSpeed = zeros(self.x_ran)
        self.yawSpeed = zeros(self.x_ran)
        self.xaccel = zeros(self.x_ran)
        self.yaccel = zeros(self.x_ran)
        self.zaccel = zeros(self.x_ran)
        self.xspeed = zeros(self.x_ran)
        self.yspeed = zeros(self.x_ran)
        self.zspeed = zeros(self.x_ran)

        self.starttime = 0.0

    def update_data(self):
        # 1) 데이터 유무 체크: vE 기준이 가장 확실
        n_total = len(self.datahub.vE)
        if n_total == 0:
            return

        # 2) 타겟 길이 결정
        win = self.x_ran
        if n_total <= win:
            n = n_total

            # --- 각속도 ---
            self.rollSpeed[-n:]  = self.datahub.rollSpeeds[-n:]
            self.pitchSpeed[-n:] = self.datahub.pitchSpeeds[-n:]
            self.yawSpeed[-n:]   = self.datahub.yawSpeeds[-n:]

            # --- 가속도 ---
            self.xaccel[-n:] = self.datahub.Xaccels[-n:]
            self.yaccel[-n:] = self.datahub.Yaccels[-n:]
            self.zaccel[-n:] = self.datahub.Zaccels[-n:]

            # --- 속도 성분 (vE/vN/vU) ---
            self.xspeed[-n:] = self.datahub.vE[-n:]
            self.yspeed[-n:] = self.datahub.vN[-n:]
            self.zspeed[-n:] = self.datahub.vU[-n:]

            # --- 시간축 ---
            if len(self.datahub.t) >= n:
                self.time[-n:] = self.datahub.t[-n:]
            else:
                # 보조 계산(밀리초 → 초): /1000.0
                hours  = self.datahub.hours[-n:] * 3600.0
                minutes= self.datahub.mins[-n:]  * 60.0
                seconds= self.datahub.secs[-n:]
                millis = self.datahub.tenmilis[-n:] / 100.0
                t_abs = hours + minutes + seconds + millis
                t0 = (self.datahub.hours[0]*3600.0 + self.datahub.mins[0]*60.0
                    + self.datahub.secs[0] + self.datahub.tenmilis[0]/1000.0)
                self.time[-n:] = t_abs - t0

        else:
            # 최근 win개만 표시
            s = -win

            # --- 각속도 ---
            self.rollSpeed[:]  = self.datahub.rollSpeeds[s:]
            self.pitchSpeed[:] = self.datahub.pitchSpeeds[s:]
            self.yawSpeed[:]   = self.datahub.yawSpeeds[s:]

            # --- 가속도 ---
            self.xaccel[:] = self.datahub.Xaccels[s:]
            self.yaccel[:] = self.datahub.Yaccels[s:]
            self.zaccel[:] = self.datahub.Zaccels[s:]

            # --- 속도 성분 ---
            self.xspeed[:] = self.datahub.vE[s:]
            self.yspeed[:] = self.datahub.vN[s:]
            self.zspeed[:] = self.datahub.vU[s:]

            # --- 시간축 ---
            if len(self.datahub.t) >= win:
                self.time[:] = self.datahub.t[s:]
            else:
                hours  = self.datahub.hours[s:] * 3600.0
                minutes= self.datahub.mins[s:]  * 60.0
                seconds= self.datahub.secs[s:]
                millis = self.datahub.tenmilis[s:] / 100.0
                t_abs = hours + minutes + seconds + millis
                # 윈도 첫 샘플 기준 0
                self.time[:] = t_abs - t_abs[0]

        # 3) 그래프 업데이트
        self.curve_rollSpeed.setData(x=self.time, y=self.rollSpeed)
        self.curve_pitchSpeed.setData(x=self.time, y=self.pitchSpeed)
        self.curve_yawSpeed.setData(x=self.time, y=self.yawSpeed)

        self.curve_xaccel.setData(x=self.time, y=self.xaccel)
        self.curve_yaccel.setData(x=self.time, y=self.yaccel)
        self.curve_zaccel.setData(x=self.time, y=self.zaccel)

        self.curve_xspeed.setData(x=self.time, y=self.xspeed)
        self.curve_yspeed.setData(x=self.time, y=self.yspeed)
        self.curve_zspeed.setData(x=self.time, y=self.zspeed)

    def graph_clear(self):

        self.time = zeros(self.x_ran)
        self.rollSpeed = zeros(self.x_ran)
        self.pitchSpeed = zeros(self.x_ran)
        self.yawSpeed = zeros(self.x_ran)
        self.xaccel = zeros(self.x_ran)
        self.yaccel = zeros(self.x_ran)
        self.zaccel = zeros(self.x_ran)

        self.curve_rollSpeed.clear()
        self.curve_pitchSpeed.clear()
        self.curve_yawSpeed.clear()

        self.curve_xaccel.clear()
        self.curve_yaccel.clear()
        self.curve_zaccel.clear()
                
        self.curve_xspeed.clear()
        self.curve_yspeed.clear()
        self.curve_zspeed.clear()

class RealTimeENUPlot(QThread):
    def __init__(self, mainwindow, datahub,
                 tail_len: int = 2000,
                 update_ms: int = 50,
                 vel_scale: float = 0.2):
        
        super().__init__()

        self.QVector3D = QVector3D
        self.mainwindow = mainwindow
        self.datahub = datahub
        self.tail_len = tail_len
        self.update_ms = update_ms
        self.vel_scale = vel_scale

        # 내부 버퍼
        from collections import deque
        self.E_hist = deque(maxlen=tail_len)
        self.N_hist = deque(maxlen=tail_len)
        self.U_hist = deque(maxlen=tail_len)
        self._last_count = 0
        self._ref_lla = None
        self._ref_ecef = None
        self._input_mode = "auto"  # "auto" | "enu" | "ecef"

        # 뷰/아이템 (UI 스레드에서 생성)
        self.trajectory_title = QLabel(self.mainwindow.container)
        self.trajectory_title.setText("<b>&#8226; Trajectory</b>")
        self.trajectory_title.setStyleSheet("color: white;")
        self.trajectory_title.setFont(ws.font_trajectory_title)
        self.trajectory_title.setGeometry(*ws.trajectory_title_geometry)   

        self.view = gl.GLViewWidget(self.mainwindow.container)
        geom = getattr(ws, "enu3d_geometry", ws.pw_trajectory_geometry)
        self.view.setGeometry(*ws.pw_trajectory_geometry)
        self.view.setWindowTitle("Trajectory")
        self.view.setCameraPosition(distance=50)

        self._add_axes()
        self.traj_item = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=2, antialias=True)
        self.view.addItem(self.traj_item)
        self.head_item = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), size=8)
        self.view.addItem(self.head_item)
        self.vel_item = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=3, antialias=True)
        self.view.addItem(self.vel_item)

        self._add_legend()  # ENU RGB legend overlay

        # UI 스레드 타이머로 갱신(안전)
        self.timer = QTimer(self.mainwindow)
        self.timer.timeout.connect(self._update_plot)
        self.timer.start(update_ms)

        self.keep_all = True          # 전체 궤적 유지 (False면 기존 tail만 그림)
        self.max_draw_pts = 20000     # 그릴 때 최대 포인트 수 (성능 보호용)

        # 전체 궤적 누적 리스트
        self.E_all = []
        self.N_all = []
        self.U_all = []

    # ---------- 좌표 변환 ----------
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
    def _ecef_to_enu(xyz, ref_ecef, ref_lat_deg, ref_lon_deg):
        lat = np.deg2rad(ref_lat_deg); lon = np.deg2rad(ref_lon_deg)
        sl, cl = np.sin(lat), np.cos(lat)
        so, co = np.sin(lon), np.cos(lon)
        dx, dy, dz = xyz - ref_ecef
        T = np.array([[-so,          co,         0],
                      [-sl*co,   -sl*so,        cl],
                      [ cl*co,    cl*so,        sl]])
        return T @ np.array([dx, dy, dz])
    
    @staticmethod
    def _ecef_to_lla(x, y, z):
        a = 6378137.0
        f = 1/298.257223563
        e2 = f*(2-f)
        b = a*(1-f)
        ep2 = (a*a - b*b) / (b*b)

        r = np.hypot(x, y)
        if r < 1e-9:
            # Pole
            lat = np.sign(z) * np.pi/2
            lon = 0.0
            h = abs(z) - b
            return np.rad2deg(lat), np.rad2deg(lon), h

        lon = np.arctan2(y, x)

        # Initial parametric latitude
        theta = np.arctan2(z * a, r * b)
        st, ct = np.sin(theta), np.cos(theta)

        lat = np.arctan2(z + ep2 * b * st*st*st,
                         r - e2 * a * ct*ct*ct)

        sl, cl = np.sin(lat), np.cos(lat)
        N = a / np.sqrt(1 - e2 * sl*sl)
        h = r/np.cos(lat) - N

        # One Bowring correction (usually enough)
        lat_prev = lat
        N = a / np.sqrt(1 - e2 * np.sin(lat_prev)**2)
        h = r/np.cos(lat_prev) - N
        lat = np.arctan2(z, r*(1 - e2*N/(N+h)))

        return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(h)
    @staticmethod
    def _ecef_vec_to_enu(vec_xyz, ref_lat_deg, ref_lon_deg):
        lat = np.deg2rad(ref_lat_deg); lon = np.deg2rad(ref_lon_deg)
        sl, cl = np.sin(lat), np.cos(lat)
        so, co = np.sin(lon), np.cos(lon)
        T = np.array([[-so,        co,       0],
                    [-sl*co,  -sl*so,     cl],
                    [ cl*co,   cl*so,     sl]])
        return T @ np.asarray(vec_xyz, dtype=float)
    
    def _has_hub_enu(self):
        return (hasattr(self.datahub, 'e_enu') and hasattr(self.datahub, 'n_enu') and hasattr(self.datahub, 'u_enu')
                and len(self.datahub.e_enu) > 0 and len(self.datahub.n_enu) > 0 and len(self.datahub.u_enu) > 0)

    def _has_hub_enu_vel(self):
        return (hasattr(self.datahub, 'vE_enu') and hasattr(self.datahub, 'vN_enu') and hasattr(self.datahub, 'vU_enu')
                and len(self.datahub.vE_enu) > 0 and len(self.datahub.vN_enu) > 0 and len(self.datahub.vU_enu) > 0)

    def _ensure_ref_from_datahub(self):
        if self._ref_lla is not None and self._ref_ecef is not None:
            return

        if all(hasattr(self.datahub, n) for n in ('Easts','Norths','Ups')) and len(self.datahub.Easts) > 0:
            x0 = float(self.datahub.Easts[0]); y0 = float(self.datahub.Norths[0]); z0 = float(self.datahub.Ups[0])
            lat0, lon0, h0 = self._ecef_to_lla(x0, y0, z0)
            self._ref_lla = (lat0, lon0, h0)
            self._ref_ecef = np.array([x0, y0, z0], dtype=float)
            return

        self._ref_lla = None
        self._ref_ecef = None
    
    def _detect_input_mode(self):
        # 1순위: Datahub가 ENU를 이미 제공
        if self._has_hub_enu():
            return "hub_enu"

        # 2순위: Easts/Norths/Ups 가 ENU인지/ECEF인지 추정
        if not all(hasattr(self.datahub, n) for n in ('Easts','Norths','Ups')) or len(self.datahub.Easts) == 0:
            return "unknown"
        try:
            x = float(self.datahub.Easts[0]); y = float(self.datahub.Norths[0]); z = float(self.datahub.Ups[0])
            norm = np.sqrt(x*x + y*y + z*z)
            return "ecef" if norm > 1.0e6 else "enu"
        except Exception:
            return "unknown"
        
    def _available_count(self):
        # hub ENU가 있으면 그 길이에 맞춘다
        if self._has_hub_enu():
            return min(len(self.datahub.e_enu), len(self.datahub.n_enu), len(self.datahub.u_enu))
        # 없으면 기존 Easts/Norths/Ups
        if all(hasattr(self.datahub, n) for n in ('Easts','Norths','Ups')):
            return min(len(self.datahub.Easts), len(self.datahub.Norths), len(self.datahub.Ups))
        return 0

    def _get_dt(self, idx):
        try:
            h, m, s = float(self.datahub.hours[idx]), float(self.datahub.mins[idx]), float(self.datahub.secs[idx])
            tm = float(self.datahub.tenmilis[idx])*0.01
            h0, m0, s0 = float(self.datahub.hours[idx-1]), float(self.datahub.mins[idx-1]), float(self.datahub.secs[idx-1])
            tm0 = float(self.datahub.tenmilis[idx-1])*0.01
            return (h*3600+m*60+s+tm) - (h0*3600+m0*60+s0+tm0)
        except Exception:
            return None

    def _get_velocity_sample(self, idx, fallback_from_pos=None):
        # 1) hub ENU 속도 있으면 그대로 사용
        if self._has_hub_enu_vel():
            try:
                return (float(self.datahub.vE_enu[idx]),
                        float(self.datahub.vN_enu[idx]),
                        float(self.datahub.vU_enu[idx]))
            except Exception:
                pass

        # 2) 포지션 ENU로부터 미분
        if fallback_from_pos is not None and idx >= 1:
            (E, N, U) = fallback_from_pos[idx]; (E0, N0, U0) = fallback_from_pos[idx-1]
            dt = self._get_dt(idx)
            if dt and dt > 1e-6:
                return ((E-E0)/dt, (N-N0)/dt, (U-U0)/dt)

        # 3) 최후: 0
        return (0.0, 0.0, 0.0)

    def _pull_new_samples(self):
        total = self._available_count()
        if total <= self._last_count:
            return None

        # 모드 결정
        if self._input_mode == "auto":
            self._input_mode = self._detect_input_mode()

        idxs = range(self._last_count, total)
        self.ENU = []

        # A) Datahub가 ENU 제공 (권장 경로)
        if self._input_mode == "hub_enu":
            for i in idxs:
                self.ENU.append((float(self.datahub.e_enu[i]),
                                float(self.datahub.n_enu[i]),
                                float(self.datahub.u_enu[i])))

        # B) 사용자가 ENU 직접 공급 (레거시)
        elif self._input_mode == "enu":
            for i in idxs:
                self.ENU.append((float(self.datahub.Easts[i]),
                                float(self.datahub.Norths[i]),
                                float(self.datahub.Ups[i])))

        # C) ECEF → ENU 변환 (폴백)
        elif self._input_mode == "ecef":
            self._ensure_ref_from_datahub()
            if self._ref_lla is None or self._ref_ecef is None:
                return None
            lat0, lon0, _ = self._ref_lla
            for i in idxs:
                x = float(self.datahub.Easts[i]); y = float(self.datahub.Norths[i]); z = float(self.datahub.Ups[i])
                e, n, u = self._ecef_to_enu(np.array([x, y, z], float), self._ref_ecef, lat0, lon0)
                self.ENU.append((e, n, u))
        else:
            return None

        # 버퍼 적재 (tail & all)
        for E, N, U in self.ENU:
            self.E_hist.append(E); self.N_hist.append(N); self.U_hist.append(U)
            if self.keep_all:
                self.E_all.append(E); self.N_all.append(N); self.U_all.append(U)

        # 속도
        last_idx = total - 1
        vE, vN, vU = self._get_velocity_sample(last_idx, fallback_from_pos=self.ENU)

        self._last_count = total
        return (vE, vN, vU)
    
    def _prepare_pos(self):
        if self.keep_all and len(self.E_all) >= 2:
            e = np.asarray(self.E_all, dtype=float)
            n = np.asarray(self.N_all, dtype=float)
            u = np.asarray(self.U_all, dtype=float)
        else:
            if len(self.E_hist) < 2:
                return None
            e = np.asarray(self.E_hist, dtype=float)
            n = np.asarray(self.N_hist, dtype=float)
            u = np.asarray(self.U_hist, dtype=float)

        L = e.shape[0]
        if L > self.max_draw_pts:
            step = int(np.ceil(L / self.max_draw_pts))
            idx = slice(0, L, step)
            e, n, u = e[idx], n[idx], u[idx]

        return np.column_stack([e, n, u])

    def _add_axes(self):
        x_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0],[10,0,0]]), color=(1,0,0,1), width=2, antialias=True)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0],[0,10,0]]), color=(0,1,0,1), width=2, antialias=True)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0],[0,0,10]]), color=(0,0,1,1), width=2, antialias=True)
        self.view.addItem(x_axis); self.view.addItem(y_axis); self.view.addItem(z_axis)

    def _autoscale_camera(self):
        if len(self.E_hist) < 2: return
        e = np.array(self.E_hist); n = np.array(self.N_hist); u = np.array(self.U_hist)
        e_mid = (e.min()+e.max())/2.0; n_mid = (n.min()+n.max())/2.0; u_mid = (u.min()+u.max())/2.0
        max_range = max(e.max()-e.min(), n.max()-n.min(), u.max()-u.min())
        dist = max(20.0, max_range*1.5)
        self.view.opts['center'] = self.QVector3D(float(e_mid), float(n_mid), float(u_mid))
        self.view.setCameraPosition(distance=dist)

    def _autoscale_camera_with(self, pos):
        if pos is None or len(pos) < 2:
            return
        e, n, u = pos[:,0], pos[:,1], pos[:,2]
        e_mid = (e.min()+e.max())/2.0
        n_mid = (n.min()+n.max())/2.0
        u_mid = (u.min()+u.max())/2.0
        max_range = max(e.max()-e.min(), n.max()-n.min(), u.max()-u.min())
        dist = max(20.0, max_range*1.5)
        self.view.opts['center'] = self.QVector3D(float(e_mid), float(n_mid), float(u_mid))
        self.view.setCameraPosition(distance=dist)

    def _update_plot(self):
        out = self._pull_new_samples()
        if out is None:
            return
        vE, vN, vU = out

        pos = self._prepare_pos()
        if pos is None:
            return

        if len(pos) >= 2:
            self.traj_item.setData(pos=pos)

        head = pos[-1].reshape(1,3)
        self.head_item.setData(pos=head)

        vel_end = head[0] + np.array([vE, vN, vU]) * self.vel_scale
        self.vel_item.setData(pos=np.vstack([head[0], vel_end]))

        # 변경: pos 기반 오토스케일
        self._autoscale_camera_with(pos)

    def _add_legend(self):
        # GLViewWidget 위치 기준으로 좌상단에 작게 띄우기
        vx, vy, vw, vh = ws.pw_trajectory_geometry  # (x, y, w, h)
        pad = 10
        box_w, box_h = 140, 80

        self.legend_bg = QFrame(self.mainwindow.container)
        self.legend_bg.setGeometry(vx + pad, vy + pad, box_w, box_h)
        self.legend_bg.setStyleSheet("""
            QFrame {
                background-color: rgba(0,0,0,160);
                border-radius: 8px;
            }
        """)
        # 마우스 이벤트 투과 (씬 회전/줌 방해 X)
        self.legend_bg.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.legend_bg.show()

        def add_row(y, rgb, text):
            r, g, b = rgb
            sw = QFrame(self.legend_bg)
            sw.setGeometry(10, y, 16, 16)
            sw.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid rgba(255,255,255,140); border-radius: 3px;")

            lb = QLabel(text, self.legend_bg)
            lb.setGeometry(34, y-2, box_w-44, 20)
            lb.setStyleSheet("color: white;")

        # ENU with RGB
        add_row(12, (255,   0,   0), "E (East)  — Red")
        add_row(34, (  0, 255,   0), "N (North) — Green")
        add_row(56, (  0,   0, 255), "U (Up)    — Blue")

    def trajectory_clear(self):
        # tail deque만 비우고, last_count 초기화 (다음 프레임부터 새로 들어온 것만 tail에 채움)
        try:
            self.E_hist.clear(); self.N_hist.clear(); self.U_hist.clear()
        except Exception:
            from collections import deque
            self.E_hist = deque(maxlen=self.tail_len)
            self.N_hist = deque(maxlen=self.tail_len)
            self.U_hist = deque(maxlen=self.tail_len)

        self._last_count = 0
        # 보기 객체 초기화
        try: self.traj_item.setData(pos=np.zeros((2, 3)))
        except: pass
        try: self.head_item.setData(pos=np.array([[0.0, 0.0, 0.0]]))
        except: pass
        try: self.vel_item.setData(pos=np.zeros((2, 3)))
        except: pass
        # 카메라 리셋은 유지/선택
        try:
            self.view.opts['center'] = self.QVector3D(0.0, 0.0, 0.0)
            self.view.setCameraPosition(distance=50)
        except: pass

    def trajectory_hard_reset(self):
        self.trajectory_clear()
        self.E_all.clear(); self.N_all.clear(); self.U_all.clear()
        self._ref_lla = None; self._ref_ecef = None
        self._input_mode = "auto"

    # QThread 수명주기
    def run(self):
        self.exec_()   # 별도 타이머는 없지만, 일관성 있게 스레드 루프 유지

    def stop(self):
        try:
            self.timer.stop()
        except Exception:
            pass
        self.quit()
        self.wait(1000)

class MapViewer_Thread(QThread):
    def __init__(self, mainwindow, datahub):
        super().__init__()
        self.mainwindow = mainwindow
        self.datahub = datahub

        # 기준 위치(초기 맵 중심). 필요시 프로젝트 현장 좌표로 바꿔줘.
        self.ref_lat = 37.45162
        self.ref_lon = 126.65058
        self.ref_h   = 0.0

        base_dir = Path(__file__).resolve().parent
        map_path = base_dir / "map.html"
        if not map_path.exists():
            raise FileNotFoundError(f"map.html을 찾을 수 없습니다: {map_path}")

        html = map_path.read_text(encoding="utf-8")
        new_width  = f"{ws.map_geometry[2]}px"
        new_height = f"{ws.map_geometry[3]}px"
        html = html.replace("width: 576px;",  f"width: {new_width};")
        html = html.replace("height: 345px;", f"height: {new_height};")
        map_path.write_text(html, encoding="utf-8")

        self.view = QWebEngineView(self.mainwindow.container)
        self.view.setGeometry(*ws.map_geometry)
        self.view.load(QUrl.fromLocalFile(str(map_path)))
        self.view.show()

    def run(self):
        self.view.loadFinished.connect(self.on_load_finished)

    def on_load_finished(self):
        page = self.view.page()
        self.script = f"""
        var lat = {self.ref_lat};
        var lng = {self.ref_lon};
        var map = L.map("map").setView([lat,lng], 17);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri',
            maxZoom: 18,
        }}).addTo(map);
        var marker = L.marker([lat,lng]).addTo(map);
        var trigger_javascript = 0;
        function updateMarker(latnew, lngnew, trigger_python) {{
            marker.setLatLng([latnew, lngnew]);
            if(trigger_python >= 1 && trigger_javascript == 0) {{
                map.setView([latnew,lngnew], 15);
                trigger_javascript = 1;
            }}
        }}
        """
        page.runJavaScript(self.script)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_marker)
        self.timer.start(1000)

    # --- ENU(m) -> 위경도(deg) 근사 변환 (작은 영역용) ---
    @staticmethod
    def _enu_to_latlon(E, N, lat0_deg, lon0_deg):
        R = 6378137.0  # 지구 장반경 (m)
        lat0 = np.deg2rad(lat0_deg)
        dlat = (N / R) * (180.0 / np.pi)
        dlon = (E / (R * np.cos(lat0))) * (180.0 / np.pi)
        return lat0_deg + dlat, lon0_deg + dlon

    def update_marker(self):
        page = self.view.page()

        # 1) 위경도 필드가 있으면 그것부터 사용
        if hasattr(self.datahub, "latitudes") and hasattr(self.datahub, "longitudes"):
            if len(self.datahub.latitudes) == 0 or len(self.datahub.longitudes) == 0:
                return
            lat = float(self.datahub.latitudes[-1])
            lng = float(self.datahub.longitudes[-1])

        # 2) 없으면 ENU + 기준 위경도로 변환
        elif len(self.datahub.Easts) > 0 and len(self.datahub.Norths) > 0:
            E = float(self.datahub.Easts[-1])
            N = float(self.datahub.Norths[-1])
            lat, lng = self._enu_to_latlon(E, N, self.ref_lat, self.ref_lon)
        else:
            # 위치 정보가 아직 없음
            return

        trig = getattr(self.datahub, "trigger_python", 0)
        page.runJavaScript(f"updateMarker({lat:.8f}, {lng:.8f}, {int(trig)})")

class RocketViewer_Thread(QThread):
    update_signal = pyqtSignal()

    def __init__(self, mainwindow, datahub):
        super().__init__()
        self.mainwindow = mainwindow
        self.datahub = datahub
        self.pose = array([1.0, 0.0, 0.0])
        self.radius = 0.1
        self.normal = array([0.0, 0.0, 0.0])
        self.x = np.random.rand(1)
        self.y = np.random.rand(1)
        self.circle_point = np.zeros((3, 5))
        self._ref_lla = None
        self._ref_ecef = None
        self.altitude_mode = "enu"  # "enu" = 기준점 대비 상대고도(U), "msl" = 절대고도(LLA h)
        self.setup_ui()

    def setup_ui(self):
        # ▶ 중앙 컨테이너를 부모로 지정
        self.view = gl.GLViewWidget(self.mainwindow.container)
        self.view.setGeometry(*ws.model_geometry)
        self.view.setWindowTitle('Rocket Viewer')
        self.view.setCameraPosition(distance=10)

        # 축 추가
        self.add_axes()

        # 1) 스크립트 기준 폴더 구하기
        base_dir = Path(__file__).resolve().parent

        # 2) rocket.obj 경로 동적 구성 및 존재 여부 체크
        obj_path = base_dir / "rocket.obj"
        if not obj_path.exists():
            raise FileNotFoundError(f"rocket.obj을 찾을 수 없습니다: {obj_path}")

        # 3) OBJ 파일 로드 및 뷰에 추가
        self.rocket_mesh = self.load_and_display_obj(str(obj_path))
        self.view.addItem(self.rocket_mesh)

        # UI 레이블 초기화
        self.speed_label = QLabel("Speed ", self.mainwindow.container)
        self.speed_label.setGeometry(*ws.speed_label_geometry)
        self.speed_label.setStyleSheet("color: #00FF00;")
        self.speed_label.setFont(ws.font_speed_text)

        self.altitude_label = QLabel("Altitude ", self.mainwindow.container)
        self.altitude_label.setGeometry(*ws.altitude_label_geometry)
        self.altitude_label.setStyleSheet("color: #00FF00;")
        self.altitude_label.setFont(ws.font_altitude_text)

        self.roll_label = QLabel("Roll : ", self.mainwindow.container)
        self.roll_label.setGeometry(*ws.roll_label_geometry)
        self.roll_label.setStyleSheet("color: #00FF00;")
        self.roll_label.setFont(ws.font_roll_text)

        self.pitch_label = QLabel("Pitch : ", self.mainwindow.container)
        self.pitch_label.setGeometry(*ws.pitch_label_geometry)
        self.pitch_label.setStyleSheet("color: #00FF00;")
        self.pitch_label.setFont(ws.font_pitch_text)

        self.yaw_label = QLabel("Yaw : ", self.mainwindow.container)
        self.yaw_label.setGeometry(*ws.yaw_label_geometry)
        self.yaw_label.setStyleSheet("color: #00FF00;")
        self.yaw_label.setFont(ws.font_yaw_text)

        self.rollspeed_label = QLabel("Roll_speed : ", self.mainwindow.container)
        self.rollspeed_label.setGeometry(*ws.rollspeed_label_geometry)
        self.rollspeed_label.setStyleSheet("color: #00FF00;")
        self.rollspeed_label.setFont(ws.font_rollspeed_text)

        self.pitchspeed_label = QLabel("Pitch_speed : ", self.mainwindow.container)
        self.pitchspeed_label.setGeometry(*ws.pitchspeed_label_geometry)
        self.pitchspeed_label.setStyleSheet("color: #00FF00;")
        self.pitchspeed_label.setFont(ws.font_pitchspeed_text)

        self.yawspeed_label = QLabel("Yaw_speed : ", self.mainwindow.container)
        self.yawspeed_label.setGeometry(*ws.yawspeed_label_geometry)
        self.yawspeed_label.setStyleSheet("color: #00FF00;")
        self.yawspeed_label.setFont(ws.font_yawspeed_text)

        self.xacc_label = QLabel("X_acc : ", self.mainwindow.container)
        self.xacc_label.setGeometry(*ws.xacc_label_geometry)
        self.xacc_label.setStyleSheet("color: #00FF00;")
        self.xacc_label.setFont(ws.font_xacc_text)

        self.yacc_label = QLabel("Y_acc : ", self.mainwindow.container)
        self.yacc_label.setGeometry(*ws.yacc_label_geometry)
        self.yacc_label.setStyleSheet("color: #00FF00;")
        self.yacc_label.setFont(ws.font_yacc_text)

        self.zacc_label = QLabel("Z_acc : ", self.mainwindow.container)
        self.zacc_label.setGeometry(*ws.zacc_label_geometry)
        self.zacc_label.setStyleSheet("color: #00FF00;")
        self.zacc_label.setFont(ws.font_zacc_text)

        # 메인 스레드에서 타이머 설정
        self.timer = QTimer(self.mainwindow)
        self.timer.timeout.connect(self.update_pose)
        self.timer.start(10)  # 10ms마다 호출

    def add_axes(self):
        # X축 (빨간색)
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [10, 0, 0]]), color=(1, 0, 0, 1), width=2, antialias=True)
        self.view.addItem(x_axis)

        # Y축 (초록색)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 10, 0]]), color=(0, 1, 0, 1), width=2, antialias=True)
        self.view.addItem(y_axis)

        # Z축 (파란색)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 10]]), color=(0, 0, 1, 1), width=2, antialias=True)
        self.view.addItem(z_axis)

    def load_and_display_obj(self, filename):
        vertices, faces = self.load_obj(filename)

        # 로켓의 중심 계산
        centroid = np.mean(vertices, axis=0)

        # 중앙에 위치시키기 위해 로켓 메쉬를 원점으로 이동
        mesh = gl.GLMeshItem(vertexes=vertices, faces=faces, drawEdges=True, edgeColor=(1, 1, 1, 1), smooth=False)
        mesh.translate(-centroid[0], -centroid[1], -centroid[2])


        return mesh

    def load_obj(self, filename):
        vertices = []
        faces = []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):  # Vertex 데이터
                    vertices.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('f '):  # Face 데이터
                    face = [int(part.split('/')[0]) - 1 for part in line.strip().split()[1:]]
                    faces.append(face)
        return np.array(vertices), np.array(faces)

    def update_pose(self):
        def _last(arr, default=np.nan):
            try:
                return float(arr[-1]) if len(arr) > 0 else float(default)
            except Exception:
                return float(default)

        def _fmt(val, suffix=""):
            return f"{val:.2f}{suffix}" if np.isfinite(val) else "N/A"

        # 1) 오리엔테이션: Datahub가 이미 ENU 기준 Z–X–Y(roll=z, pitch=x, yaw=y)로 계산해둠
        roll_z  = _last(self.datahub.rolls)   # deg
        pitch_x = _last(self.datahub.pitchs)  # deg
        yaw_y   = _last(self.datahub.yaws)    # deg

        self.rocket_mesh.resetTransform()
        # 적용 순서/축: Z(roll) → X(pitch) → Y(yaw)
        self.rocket_mesh.rotate(roll_z,  0, 0, 1)  # roll about Z
        self.rocket_mesh.rotate(pitch_x, 1, 0, 0)  # pitch about X
        self.rocket_mesh.rotate(yaw_y,   0, 1, 0)  # yaw about Y

        # 2) 속도: 있으면 v*_enu 사용(크기는 좌표계와 무관하지만 일관성 위해 ENU 우선)
        if hasattr(self.datahub, 'vE_enu') and len(self.datahub.vE_enu) > 0:
            vE = _last(self.datahub.vE_enu)
            vN = _last(self.datahub.vN_enu)
            vU = _last(self.datahub.vU_enu)
            spd = (vE**2 + vN**2 + vU**2) ** 0.5
        elif len(self.datahub.speed) > 0:
            spd = _last(self.datahub.speed)
        else:
            # ECEF 속도로 대체
            vE = _last(self.datahub.vE)
            vN = _last(self.datahub.vN)
            vU = _last(self.datahub.vU)
            spd = (vE**2 + vN**2 + vU**2) ** 0.5

        self.speed_label.setText(f"Speed {_fmt(spd,'m/s')}")

        alt = _last(self.datahub.u_enu)
        self.altitude_label.setText(f"Altitude {_fmt(alt,'m')}")

        # 4) 각도 레이블(이미 deg): Datahub 계산값 그대로 사용
        self.roll_label.setText(f"Roll : {_fmt(roll_z,'°')}")
        self.pitch_label.setText(f"Pitch : {_fmt(pitch_x,'°')}")
        self.yaw_label.setText(f"Yaw : {_fmt(yaw_y,'°')}")

        # 5) 각속도/가속도
        self.rollspeed_label.setText(f"Roll_speed : {_fmt(_last(self.datahub.rollSpeeds),'Rad/s')}")
        self.pitchspeed_label.setText(f"Pitch_speed : {_fmt(_last(self.datahub.pitchSpeeds),'Rad/s')}")
        self.yawspeed_label.setText(f"Yaw_speed : {_fmt(_last(self.datahub.yawSpeeds),'Rad/s')}")

        self.xacc_label.setText(f"X_acc : {_fmt(_last(self.datahub.Xaccels),'m/s²')}")
        self.yacc_label.setText(f"Y_acc : {_fmt(_last(self.datahub.Yaccels),'m/s²')}")
        self.zacc_label.setText(f"Z_acc : {_fmt(_last(self.datahub.Zaccels),'m/s²')}")


    def data_label_clear(self):
        """모든 데이터 레이블을 초기 상태로 리셋"""
        # 속도/고도
        self.speed_label.setText("Speed 0.00 m/s")
        self.altitude_label.setText("Altitude 0.00 m")

        # 오일러 각
        self.roll_label.setText("Roll : 0.00°")
        self.pitch_label.setText("Pitch : 0.00°")
        self.yaw_label.setText("Yaw : 0.00°")

        # 각속도
        self.rollspeed_label.setText("Roll_speed : 0.00 Rad/s")
        self.pitchspeed_label.setText("Pitch_speed : 0.00 Rad/s")
        self.yawspeed_label.setText("Yaw_speed : 0.00 Rad/s")

        # 가속도
        self.xacc_label.setText("X_acc : 0.00 m/s²")
        self.yacc_label.setText("Y_acc : 0.00 m/s²")
        self.zacc_label.setText("Z_acc : 0.00 m/s²")


    def run(self):
        self.exec_()  # QThread에서 이벤트 루프를 실행

class MainWindow(PageWindow):
    def __init__(self, datahub):
        super().__init__()
        self.datahub = datahub

        # ▶ 중앙 컨테이너 생성
        self.container = QWidget(self)
        self.setCentralWidget(self.container)

        # ▶ 동적 기준 디렉터리 설정 (__file__ 이 있는 폴더)
        base_dir = Path(__file__).resolve().parent
        self.dir_path = str(base_dir)     # 문자열이 필요하면 str(), 아니면 Path 객체 그대로 써도 됩니다

        self.initUI()
        self.initGraph()

        """Start Thread"""
        self.mapviewer = MapViewer_Thread(self,datahub)
        self.graphviewer = GraphViewer_Thread(self,datahub)
        self.rocketviewer = RocketViewer_Thread(self,datahub)
        self.enuviewer = RealTimeENUPlot(self, datahub)  # ENU 3D 뷰어

        self.initMenubar()

        self.enuviewer.start()
        self.mapviewer.start()
        self.graphviewer.start()
        self.rocketviewer.start()

        self.resetcheck = 0

        self.csv_player = None
        self.replay_csv_path = None  # 사용자가 고를 CSV 경로
        
    def initUI(self):

        """Set Buttons"""
        self.start_button = QPushButton("Start",self.container)
        self.stop_button = QPushButton("Stop",self.container)
        self.reset_button = QPushButton("Reset",self.container)

        self.launch1_button = QPushButton("Launch_1",self.container)
        self.launch2_button = QPushButton("Launch_2",self.container)
        self.launch_stop_button = QPushButton("Launch\nStop",self.container)
        self.emergency_parachute_button = QPushButton("Emer\nParachute",self.container)
        self.staging_stop_button = QPushButton("Staging\nStop",self.container)
        self.emergency_staging_button = QPushButton("Emer\nStagigng",self.container)
        self.nc1_button = QPushButton("NC_1",self.container)
        self.nc2_button = QPushButton("NC_2",self.container)
        self.nc3_button = QPushButton("NC_3",self.container)

        self.now_status = QLabel(ws.wait_status,self.container)
        self.rf_port_edit = QLineEdit("COM8",self.container)
        self.port_text = QLabel("Port:",self.container)
        self.baudrate_edit = QLineEdit("115200",self.container)
        self.baudrate_text = QLabel("Baudrate:",self.container)
        self.guide_text = QLabel(ws.guide,self.container)
        self.port_text.setStyleSheet("color: white;")
        self.baudrate_text.setStyleSheet("color: white;")

        self.start_button.setFont(ws.font_start_text)
        self.stop_button.setFont(ws.font_stop_text)
        self.reset_button.setFont(ws.font_reset_text)

        self.launch1_button.setFont(ws.font_button_text)
        self.launch2_button.setFont(ws.font_button_text)
        self.launch_stop_button.setFont(ws.font_button_text)
        self.emergency_parachute_button.setFont(ws.font_button_text)
        self.staging_stop_button.setFont(ws.font_button_text)
        self.emergency_staging_button.setFont(ws.font_button_text)
        self.nc1_button.setFont(ws.font_button_text)
        self.nc2_button.setFont(ws.font_button_text)
        self.nc3_button.setFont(ws.font_button_text)

        self.rf_port_edit.setStyleSheet("background-color: rgb(255,255,255);")
        self.baudrate_edit.setStyleSheet("background-color: rgb(255,255,255);")
        self.start_button.setStyleSheet("background-color: rgb(200,0,0); color: rgb(250, 250, 250);font-weight: bold; font-weight: bold; border-radius: 25px;")
        self.stop_button.setStyleSheet("background-color: rgb(0,0,139); color: rgb(250, 250, 250);font-weight: bold; font-weight: bold; border-radius: 25px;")
        self.reset_button.setStyleSheet("background-color: rgb(120,120,140); color: rgb(250, 250, 250);font-weight: bold; font-weight: bold; border-radius: 25px;")

        self.shadow_start_button = QGraphicsDropShadowEffect()
        self.shadow_stop_button = QGraphicsDropShadowEffect()
        self.shadow_reset_button = QGraphicsDropShadowEffect()
        self.shadow_start_button.setOffset(6)
        self.shadow_stop_button.setOffset(6)
        self.shadow_reset_button.setOffset(6)
        self.start_button.setGraphicsEffect(self.shadow_start_button)
        self.stop_button.setGraphicsEffect(self.shadow_stop_button)
        self.reset_button.setGraphicsEffect(self.shadow_reset_button)

        self.baudrate_text.setFont(ws.font_baudrate)
        self.port_text.setFont(ws.font_portText)
        self.guide_text.setFont(ws.font_guideText)

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.rf_port_edit.setEnabled(True)
        self.baudrate_edit.setEnabled(True)

        """Set Buttons Connection"""
        self.start_button.clicked.connect(self.start_button_clicked)
        self.stop_button.clicked.connect(self.stop_button_clicked)
        self.reset_button.clicked.connect(self.reset_button_clicked)

        """Set Geometry"""
        self.start_button.setGeometry(*ws.start_geometry)
        self.stop_button.setGeometry(*ws.stop_geometry)
        self.reset_button.setGeometry(*ws.reset_geometry)

        self.launch1_button.setGeometry(*ws.launch1_geometry)
        self.launch2_button.setGeometry(*ws.launch2_geometry)
        self.launch_stop_button.setGeometry(*ws.launch_stop_geometry)
        self.emergency_parachute_button.setGeometry(*ws.emergency_parachute_geometry)
        self.staging_stop_button.setGeometry(*ws.staging_stop_geometry)
        self.emergency_staging_button.setGeometry(*ws.emergency_staging_geometry)
        self.nc1_button.setGeometry(*ws.nc1_geometry)
        self.nc2_button.setGeometry(*ws.nc2_geometry)
        self.nc3_button.setGeometry(*ws.nc3_geometry)

        self._toggle_labels = {}
        def setup_toggle(btn, on_text, off_text, initial=False):
            base = btn.text()
            self._toggle_labels[btn] = (f"{base}: {on_text}", f"{base}: {off_text}")
            btn.setCheckable(True)
            btn.setChecked(initial)
            on_label, off_label = self._toggle_labels[btn]
            btn.setText(on_label if initial else off_label)
            self._apply_toggle_style(btn, initial)
            btn.toggled.connect(lambda checked, b=btn: self.on_toggle(b, checked))

        # 라벨은 용도 맞춰 커스텀(필요시 수정 가능)
        setup_toggle(self.launch1_button,            on_text="\nARMED",  off_text="\nSAFE",   initial=False)
        setup_toggle(self.launch2_button,            on_text="\nARMED",  off_text="\nSAFE",   initial=False)
        setup_toggle(self.launch_stop_button,        on_text="\nACTIVE", off_text="\nIDLE",   initial=False)
        setup_toggle(self.emergency_parachute_button,on_text="\nREADY",  off_text="\nIDLE",   initial=False)
        setup_toggle(self.staging_stop_button,       on_text="\nACTIVE", off_text="\nIDLE",   initial=False)
        setup_toggle(self.emergency_staging_button,  on_text="\nREADY",  off_text="\nIDLE",   initial=False)
        setup_toggle(self.nc1_button,                on_text="\nON",     off_text="\nOFF",    initial=False)
        setup_toggle(self.nc2_button,                on_text="\nON",     off_text="\nOFF",    initial=False)
        setup_toggle(self.nc3_button,                on_text="\nON",     off_text="\nOFF",    initial=False)

        self.port_text.setGeometry(*ws.port_text_geometry)
        self.rf_port_edit.setGeometry(*ws.port_edit_geometry)
        self.baudrate_text.setGeometry(*ws.baudrate_text_geometry)
        self.baudrate_edit.setGeometry(*ws.baudrate_edit_geometry)
        self.guide_text.setGeometry(*ws.cmd_geometry)
        self.now_status.setGeometry(*ws.status_geometry)
        self.now_status.setFont(ws.font_status_text)
        self.now_status.setStyleSheet("color:#00FF00;")
        
        base_dir = Path(__file__).resolve().parent

        # team logo (부모를 self.container 로 변경)
        logo_path = base_dir / 'team_logo.png'
        self.team_logo = QLabel(self.container)
        self.team_logo.setPixmap(
            QPixmap(str(logo_path))
            .scaled(*ws.team_logo_geometry[2:4], Qt.KeepAspectRatio)
        )
        self.team_logo.setGeometry(*ws.team_logo_geometry)

        # irri logo
        irri_path = base_dir / 'irri.png'
        self.irri_logo = QLabel(self.container)
        self.irri_logo.setPixmap(
            QPixmap(str(irri_path))
            .scaled(*ws.irri_logo_geometry[2:4], Qt.KeepAspectRatio)
        )
        self.irri_logo.setGeometry(*ws.irri_logo_geometry)

        # patch logo
        patch_path = base_dir / 'patch.png'
        self.patch_logo = QLabel(self.container)
        self.patch_logo.setPixmap(
            QPixmap(str(patch_path))
            .scaled(*ws.patch_logo_geometry[2:4], Qt.KeepAspectRatio)
        )
        self.patch_logo.setGeometry(*ws.patch_logo_geometry)

        # patch22 logo
        patch22_path = base_dir / '22patch.png'
        self.patch22_logo = QLabel(self.container)
        self.patch22_logo.setPixmap(
            QPixmap(str(patch22_path))
            .scaled(*ws.patch22_logo_geometry[2:4], Qt.KeepAspectRatio)
        )
        self.patch22_logo.setGeometry(*ws.patch22_logo_geometry)

        # patch24 logo
        patch24_path = base_dir / '24patch.png'
        self.patch24_logo = QLabel(self.container)
        self.patch24_logo.setPixmap(
            QPixmap(str(patch24_path))
            .scaled(*ws.patch24_logo_geometry[2:4], Qt.KeepAspectRatio)
        )
        self.patch24_logo.setGeometry(*ws.patch24_logo_geometry)
    
    #상단 메뉴바
    def initMenubar(self):
        self.statusBar()

        change_Action = QAction('Analysis', self.container)
        change_Action.setShortcut('Ctrl+L')
        change_Action.triggered.connect(self.gosub)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('Menu')
        filemenu.addAction(change_Action)
        # 스타일시트 수정
        self.setStyleSheet("""
            QMenuBar {
                background-color: rgb(50,50,50);
                color: rgb(255,255,255);
                border: 1px solid rgb(50,50,50);
            }
            QMenu {
                background-color: rgb(255,255,255);  /* 메뉴 드롭다운 창의 배경색 */
                color: rgb(0,0,0);          /* 메뉴 항목의 글씨 색 */
                border: 1px solid rgb(50,50,50);  /* 메뉴 드롭다운 창의 테두리 색 */
            }
            QMenu::item::selected {
                background-color: rgb(50,50,50);  /* 선택된 메뉴 항목의 배경색 */
                color: rgb(255,255,255);          /* 선택된 메뉴 항목의 글씨 색 */
            }
        """)

    def initGraph(self):
        self.xspeed_hide_checkbox = QCheckBox("v_x", self.container)
        self.xspeed_hide_checkbox.setStyleSheet("color: white;")

        self.yspeed_hide_checkbox = QCheckBox("v_y", self.container)
        self.yspeed_hide_checkbox.setStyleSheet("color: white;")

        self.zspeed_hide_checkbox = QCheckBox("v_z", self.container)
        self.zspeed_hide_checkbox.setStyleSheet("color: white;")

        self.rollspeed_hide_checkbox = QCheckBox("w_x",self.container)
        self.rollspeed_hide_checkbox.setStyleSheet("color: white;")
        
        self.pitchspeed_hide_checkbox = QCheckBox("w_y",self.container)
        self.pitchspeed_hide_checkbox.setStyleSheet("color: white;")
        
        self.yawspeed_hide_checkbox = QCheckBox("w_z",self.container)
        self.yawspeed_hide_checkbox.setStyleSheet("color: white;")

        self.xacc_hide_checkbox = QCheckBox("a_x",self.container)
        self.xacc_hide_checkbox.setStyleSheet("color: white;")
        
        self.yacc_hide_checkbox = QCheckBox("a_y",self.container)
        self.yacc_hide_checkbox.setStyleSheet("color: white;")
        
        self.zacc_hide_checkbox = QCheckBox("a_z",self.container)
        self.zacc_hide_checkbox.setStyleSheet("color: white;")

        self.xspeed_hide_checkbox.setGeometry(*ws.vx_checker_geometry)
        self.yspeed_hide_checkbox.setGeometry(*ws.vy_checker_geometry)
        self.zspeed_hide_checkbox.setGeometry(*ws.vz_checker_geometry)

        self.rollspeed_hide_checkbox.setGeometry(*ws.rollS_checker_geomoetry)
        self.pitchspeed_hide_checkbox.setGeometry(*ws.pitchS_checker_geomoetry)
        self.yawspeed_hide_checkbox.setGeometry(*ws.yawS_checker_geomoetry)

        self.xacc_hide_checkbox.setGeometry(*ws.ax_checker_geomoetry)
        self.yacc_hide_checkbox.setGeometry(*ws.ay_checker_geomoetry)
        self.zacc_hide_checkbox.setGeometry(*ws.az_checker_geomoetry)

        self.xacc_hide_checkbox.setFont(ws.checker_font)
        self.yacc_hide_checkbox.setFont(ws.checker_font)
        self.zacc_hide_checkbox.setFont(ws.checker_font)

        self.xspeed_hide_checkbox.stateChanged.connect(self.xspeed_hide_checkbox_state)
        self.yspeed_hide_checkbox.stateChanged.connect(self.yspeed_hide_checkbox_state)
        self.zspeed_hide_checkbox.stateChanged.connect(self.zspeed_hide_checkbox_state)
        self.rollspeed_hide_checkbox.stateChanged.connect(self.rollspeed_hide_checkbox_state)
        self.pitchspeed_hide_checkbox.stateChanged.connect(self.pitchspeed_hide_checkbox_state)
        self.yawspeed_hide_checkbox.stateChanged.connect(self.yawspeed_hide_checkbox_state)
        self.xacc_hide_checkbox.stateChanged.connect(self.xacc_hide_checkbox_state)
        self.yacc_hide_checkbox.stateChanged.connect(self.yacc_hide_checkbox_state)
        self.zacc_hide_checkbox.stateChanged.connect(self.zacc_hide_checkbox_state)

        self.xspeed_hide_checkbox.setFont(ws.checker_font)
        self.yspeed_hide_checkbox.setFont(ws.checker_font)
        self.zspeed_hide_checkbox.setFont(ws.checker_font)
        self.rollspeed_hide_checkbox.setFont(ws.checker_font)
        self.pitchspeed_hide_checkbox.setFont(ws.checker_font)
        self.yawspeed_hide_checkbox.setFont(ws.checker_font)

    def gosub(self):
        self.goto("sub")

    def _apply_toggle_style(self, btn, checked: bool):
        if checked:
            btn.setStyleSheet(
                "background-color: rgb(20,120,40); color: white; font-weight: bold; border-radius: 12px;"
            )
        else:
            btn.setStyleSheet(
                "background-color: rgb(60,60,70); color: white; border-radius: 12px;"
            )
    
    def on_toggle(self, btn, checked: bool):
        """토글 상태 변경 시 라벨/스타일 갱신 + 버튼 상태를 button_data(1byte)에 반영"""
        # 1) 라벨/스타일 갱신
        on_label, off_label = self._toggle_labels[btn]
        btn.setText(on_label if checked else off_label)
        self._apply_toggle_style(btn, checked)

        # 2) 현재 UI의 토글 상태들로부터 '새 버튼 바이트'를 완전히 재계산
        new_val = 0

        # bit0: launch1 & launch2 가 모두 ON일 때 1
        if self.launch1_button.isChecked() and self.launch2_button.isChecked():
            new_val |= (1 << 0)

        # 나머지 비트: 개별 토글의 현재 상태를 그대로 반영
        if self.launch_stop_button.isChecked():
            new_val |= (1 << 1)
        if self.emergency_parachute_button.isChecked():
            new_val |= (1 << 2)
        if self.staging_stop_button.isChecked():
            new_val |= (1 << 3)
        if self.emergency_staging_button.isChecked():
            new_val |= (1 << 4)
        if self.nc1_button.isChecked():
            new_val |= (1 << 5)
        if self.nc2_button.isChecked():
            new_val |= (1 << 6)
        if self.nc3_button.isChecked():
            new_val |= (1 << 7)

        # 3) 최신 값 저장 (변화가 있을 때만 append 하고 싶다면 if 조건 추가)
        # if self.datahub.button_data.size == 0 or int(self.datahub.button_data[-1]) != new_val:
        self.datahub.button_data = np.append(self.datahub.button_data, np.uint8(new_val))

    # Run when start button is clicked
    def start_button_clicked(self):
        if self.resetcheck == 0:
            self.datahub.clear()
            
            # 스타일 시트를 이용해 배경색과 글씨 색 모두 설정
            self.setStyleSheet("""
                QMessageBox {
                    background-color: black;
                    color: white;
                }
                QLabel {
                    background-color: black;           
                    color: white;
                }
                QPushButton {
                    background-color: white;
                    color: black;
                }
                QInputDialog {
                    background-color: black;
                    color: black;
                }
                QLineEdit {
                    background-color: white;  /* QLineEdit의 배경색을 흰색으로 설정 */
                    color: black;  /* 입력 글씨 색상을 화이트로 설정 */
                }
             """)

            msg_box = QMessageBox(self)
            msg_box.setStyleSheet("QMessageBox {background-color: white;}")
            msg_box.information(self, "information", "Program Start")

            input_dialog = QInputDialog(self)
            input_dialog.setStyleSheet("QInputDialog {background-color: white;}")
            FileName, ok = input_dialog.getText(
                self, 'Input Dialog', 'Enter your File Name', QLineEdit.Normal, "Your File Name"
            )
            file_dir = os.path.join(os.path.dirname(self.dir_path), "log")
            os.makedirs(file_dir, exist_ok=True)
            file_path = os.path.join(file_dir, FileName) + ".csv"

            port_text = self.rf_port_edit.text().strip().upper()

            if port_text == "CSV":
                # 파일 선택 (로그 폴더 기본, 없다면 현재 폴더)
                start_dir = os.path.join(dirname(self.dir_path), "log")
                if not os.path.isdir(start_dir):
                    start_dir = dirname(self.dir_path)

                csv_path, _ = QFileDialog.getOpenFileName(
                    self, "Select CSV to Replay", start_dir, "CSV Files (*.csv);;All Files (*)"
                )
                if not csv_path:
                    return  # 사용자 취소

                self.replay_csv_path = csv_path
                # 통신 시작 플래그 등 UI 상태 업데이트만 동일하게
                self.datahub.communication_start()
                self.datahub.datasaver_start()
                self.now_status.setText(ws.start_status)
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.rf_port_edit.setEnabled(False)
                self.baudrate_edit.setEnabled(False)
                self.shadow_start_button.setOffset(0)
                self.shadow_stop_button.setOffset(6)
                self.shadow_reset_button.setOffset(6)

                # CSVPlayer 시작 & 그래프 업데이트 연결
                try:
                    self.csv_player = CSVPlayer(self.replay_csv_path, self.datahub, hz=5.0, parent=self)
                    # 새 샘플 마다 2D 그래프 즉시 갱신 (3D/맵은 자체 타이머로 갱신)
                    self.csv_player.sampleReady.connect(self.graphviewer.update_data)
                    self.csv_player.start()
                except Exception as e:
                    print(f"[CSVPlayer] start error: {e}")
                    QMessageBox.warning(self, "warning", "CSV 재생 시작 실패")
                    # UI 복구
                    self.datahub.communication_stop()
                    self.datahub.datasaver_stop()
                    self.start_button.setEnabled(True)
                    self.stop_button.setEnabled(False)
                    self.rf_port_edit.setEnabled(True)
                    self.baudrate_edit.setEnabled(True)
                return  # CSV 모드 처리는 여기서 종료

            else:
                if os.path.exists(file_path):
                    QMessageBox.information(self, "information", "Same file already exist")

                if ok:
                    self.datahub.mySerialPort = self.rf_port_edit.text()
                    self.datahub.myBaudrate   = self.baudrate_edit.text()
                    self.datahub.file_Name    = FileName + '.csv'

                    # 1) 통신 시작
                    self.datahub.serial_port_error = -1
                    self.datahub.communication_start()

                    # 2) "알 수 없음" 상태로 초기화 후 Receiver의 판정 기다리기
                    t0 = time.perf_counter()
                    timeout_s = 2.5
                    while (self.datahub.serial_port_error == -1
                        and (time.perf_counter() - t0) < timeout_s):
                        QApplication.processEvents()
                        time.sleep(0.01)

                    status = self.datahub.serial_port_error

                    if status != 0:
                        # 실패(1) 또는 타임아웃(-1 그대로) → 에러 처리
                        QMessageBox.warning(self, "warning", "Check the Port or Baudrate again.")
                        self.datahub.communication_stop()
                        # UI 복구
                        self.start_button.setEnabled(True)
                        self.stop_button.setEnabled(False)
                        self.rf_port_edit.setEnabled(True)
                        self.baudrate_edit.setEnabled(True)
                    else:
                        # 성공
                        self.datahub.datasaver_start()
                        self.now_status.setText(ws.start_status)
                        self.start_button.setEnabled(False)
                        self.stop_button.setEnabled(True)
                        self.rf_port_edit.setEnabled(False)
                        self.baudrate_edit.setEnabled(False)
                        self.shadow_start_button.setOffset(0)
                        self.shadow_stop_button.setOffset(6)
                        self.shadow_reset_button.setOffset(6)

        else:
            # reset 후 재시작 분기도 동일하게 대기/판정 처리 권장
            self.datahub.communication_start()
            self.datahub.serial_port_error = -1
            t0 = time.perf_counter()
            timeout_s = 2.5
            while (self.datahub.serial_port_error == -1
                and (time.perf_counter() - t0) < timeout_s):
                QApplication.processEvents()
                time.sleep(0.01)

            status = self.datahub.serial_port_error
            if status != 0:
                QMessageBox.warning(self, "warning", "Check the Port or Baudrate again.")
                self.datahub.communication_stop()
                return

            self.now_status.setText(ws.start_status)
            self.now_status.setStyleSheet("color:#00FF00;")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.rf_port_edit.setEnabled(False)
            self.baudrate_edit.setEnabled(False)
            self.shadow_start_button.setOffset(0)
            self.shadow_stop_button.setOffset(6)
            self.shadow_reset_button.setOffset(6)
            self.resetcheck = 0

    # Run when stop button is clicked
    def stop_button_clicked(self):
        # CSV 모드라면 재생 중지
        if self.csv_player is not None:
            try:
                self.csv_player.stop()
                self.csv_player.wait(1000)
            except Exception:
                pass
            self.csv_player = None

        self.datahub.communication_stop()
        self.now_status.setText(ws.stop_status)
        self.now_status.setStyleSheet("color:#00FF00;")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.rf_port_edit.setEnabled(True)
        self.baudrate_edit.setEnabled(True)
        self.shadow_start_button.setOffset(6)
        self.shadow_stop_button.setOffset(0)
        self.shadow_reset_button.setOffset(6)
        self.resetcheck = 1

    def reset_button_clicked(self):
        request = QMessageBox.question(self,'Message', 'Are you sure to reset?')
        if request == QMessageBox.Yes:
            # CSV 모드라면 재생 중지
            if self.csv_player is not None:
                try:
                    self.csv_player.stop()
                    self.csv_player.wait(1000)
                except Exception:
                    pass
            self.csv_player = None

            msg_box = QMessageBox(self)
            msg_box.setStyleSheet("QMessageBox {background-color: white;}")
            msg_box.information(self,"information","Program Reset")
            self.datahub.communication_stop()
            self.datahub.datasaver_stop()
            self.now_status.setText(ws.wait_status)
            self.now_status.setStyleSheet("color:#00FF00;")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.rf_port_edit.setEnabled(False)
            self.shadow_start_button.setOffset(6)
            self.shadow_stop_button.setOffset(0)
            self.shadow_reset_button.setOffset(0)
            self.graphviewer.graph_clear()
            self.enuviewer.trajectory_clear()
            self.rocketviewer.data_label_clear()
            self.datahub.clear()
            self.resetcheck = 0
        else:
            msg_box = QMessageBox(self)
            msg_box.setStyleSheet("QMessageBox {background-color: white;}")
            msg_box.information(self,"information","Cancel")


    #curve hide check box is clicked
    def xspeed_hide_checkbox_state(self,state):
        self.graphviewer.curve_xspeed.setVisible(state != Qt.Checked)
    def yspeed_hide_checkbox_state(self,state):
        self.graphviewer.curve_yspeed.setVisible(state != Qt.Checked)
    def zspeed_hide_checkbox_state(self,state):
        self.graphviewer.curve_zspeed.setVisible(state != Qt.Checked)        
    def rollspeed_hide_checkbox_state(self,state):
        self.graphviewer.curve_rollSpeed.setVisible(state != Qt.Checked)
    def pitchspeed_hide_checkbox_state(self,state):
        self.graphviewer.curve_pitchSpeed.setVisible(state != Qt.Checked)
    def yawspeed_hide_checkbox_state(self,state):
        self.graphviewer.curve_yawSpeed.setVisible(state != Qt.Checked)
    def xacc_hide_checkbox_state(self,state):
        self.graphviewer.curve_xaccel.setVisible(state != Qt.Checked)
    def yacc_hide_checkbox_state(self,state):
        self.graphviewer.curve_yaccel.setVisible(state != Qt.Checked)
    def zacc_hide_checkbox_state(self,state):
        self.graphviewer.curve_zaccel.setVisible(state != Qt.Checked)

    def _install_shortcuts_clicked(self):
        """단축키로 실제 버튼 .click()을 호출한다.
        → 마우스 클릭과 동일하게 clicked/toggled 시그널이 흐름."""
        self._shortcuts = []  # 가비지 컬렉션 방지용 보관

        # 단축키 → 버튼.click 매핑
        mapping = {
            "F5":   self.start_button.click,       # Start
            "F6":   self.stop_button.click,        # Stop
            "F9":   self.reset_button.click,       # Reset

            "Alt+1": self.launch1_button.click,    # Launch_1 (토글)
            "Alt+2": self.launch2_button.click,    # Launch_2 (토글)
            "Alt+3": self.launch_stop_button.click,# Launch_Stop (토글)

            "Alt+4": self.emergency_parachute_button.click,  # Emer_Parachute (토글)
            "Alt+5": self.staging_stop_button.click,         # Staging_Stop (토글)
            "Alt+6": self.emergency_staging_button.click,    # Emer_Stagigng (토글)

            "Alt+7": self.nc1_button.click,        # NC_1 (토글)
            "Alt+8": self.nc2_button.click,        # NC_2 (토글)
            "Alt+9": self.nc3_button.click,        # NC_3 (토글)
        }

        for seq, func in mapping.items():
            sc = QShortcut(QKeySequence(seq), self)
            # 페이지 어디에 포커스가 있어도 동작하게 전역 컨텍스트로
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(func)
            self._shortcuts.append(sc)

        # 각 버튼 툴팁에 단축키 표시(선택)
        for seq, func in mapping.items():
            # func는 button.click이므로, button 객체를 다시 얻어온다
            # (람다/partial로 버튼을 같이 보관해도 좋지만, 여기선 간단히 매핑 한번 더 해줌)
            pass

        button_by_seq = {
            "F5": self.start_button, "F6": self.stop_button, "F9": self.reset_button,
            "Alt+1": self.launch1_button, "Alt+2": self.launch2_button, "Alt+0": self.launch_stop_button,
            "Alt+P": self.emergency_parachute_button, "Alt+G": self.staging_stop_button, "Alt+E": self.emergency_staging_button,
            "Alt+Z": self.nc1_button, "Alt+X": self.nc2_button, "Alt+C": self.nc3_button,
        }
        for seq, btn in button_by_seq.items():
            btn.setToolTip((btn.toolTip() + " | " if btn.toolTip() else "") + f"Shortcut: {seq}")

class CSVPlayer(QThread):
    sampleReady = pyqtSignal()

    def __init__(self, csv_path, datahub, hz=5.0, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.datahub = datahub
        self.hz = float(hz)
        self._running = True
        self._paused = False

    def stop(self):
        self._running = False

    def pause(self, yes=True):
        self._paused = yes

    def run(self):
        df = pd.read_csv(self.csv_path, header=None, names=COLUMN_NAMES)
        dt = 1.0 / max(1e-6, self.hz)
        next_t = time.perf_counter()

        for _, row in df.iterrows():
            if not self._running:
                break

            # pause
            while self._paused and self._running:
                time.sleep(0.05)

            # push one sample
            self.datahub.update_from_row(row.values.tolist())

            # tell UI “new sample”
            self.sampleReady.emit()

            # pacing ~ 5 Hz
            next_t += dt
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # if lagging, reset target to now
                next_t = time.perf_counter()

class TimeAxisItem(AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLabel(text='Time', units=None)
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):
        return [str(timedelta(milliseconds = millis))[:-4] for millis in values]

#  Analysis1
class SubWindow(PageWindow):
    def __init__(self,datahub):
        self.log_dir = 'log'  # log 폴더 경로 설정
        super().__init__()
        
        self.datahub = datahub
        self.timespace = None
        self.initUI()
        self.initGraph()
        self.initMenubar()

    def initUI(self):
        self.csv_name_edit = QLineEdit("{}".format(self.datahub.file_Name),self)
        self.analysis_button = QPushButton("Analysis", self)
        self.analysis_angular_button = QPushButton("Angular Data Analysis", self)
        self.analysis_alnsp_button = QPushButton("Altitude & Speed Analysis", self)
        self.set_range_button = QPushButton("Reset range", self)
        self.max_altitude_label = QLabel("Max. altitude",self)
        self.max_speed_label = QLabel("Max. speed",self)
        self.max_accel_label = QLabel("Max. accel", self)

        self.max_altitude = QLabel("0 ",self)
        self.max_speed = QLabel("0 ",self)
        self.max_accel = QLabel("0 ", self)

        self.max_altitude_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        self.max_speed_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        self.max_accel_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        self.max_altitude.setStyleSheet("font-weight: bold; color: #00ff00;")
        self.max_speed.setStyleSheet("font-weight: bold; color: #00ff00;")
        self.max_accel.setStyleSheet("font-weight: bold; color: #00ff00;")
        self.analysis_button.setStyleSheet("background-color: rgb(120,120,140); font-weight: bold; border-radius: 25px; color: #ffffff;")
        self.analysis_angular_button.setStyleSheet("background-color: rgb(120,120,140); font-weight: bold; border-radius: 25px; color: #ffffff;")
        self.analysis_alnsp_button.setStyleSheet("background-color: rgb(120,120,140); font-weight: bold; border-radius: 25px; color: #ffffff;")
        self.set_range_button.setStyleSheet("background-color: rgb(120,120,140); font-weight: bold; border-radius: 25px; color: #ffffff;")
        self.csv_name_edit.setStyleSheet("background-color: rgb(250,250,250); color: black;")

        self.shadow_analysis_button = QGraphicsDropShadowEffect()
        self.shadow_analysis_angular_button = QGraphicsDropShadowEffect()
        self.shadow_analysis_alnsp_button = QGraphicsDropShadowEffect()
        self.shadow_set_range_button = QGraphicsDropShadowEffect()
        self.shadow_analysis_button.setOffset(6)
        self.shadow_analysis_angular_button.setOffset(6)
        self.shadow_analysis_alnsp_button.setOffset(6)
        self.shadow_set_range_button.setOffset(6)
        self.analysis_button.setGraphicsEffect(self.shadow_analysis_button)
        self.analysis_angular_button.setGraphicsEffect(self.shadow_analysis_angular_button)
        self.analysis_alnsp_button.setGraphicsEffect(self.shadow_analysis_alnsp_button)
        self.set_range_button.setGraphicsEffect(self.shadow_set_range_button)

        self.csv_name_edit.setGeometry(*ws.csv_name_geometry)
        self.analysis_button.setGeometry(*ws.analysis_button_geometry)
        self.analysis_angular_button.setGeometry(*ws.analysis_angular_button_geometry)
        self.analysis_alnsp_button.setGeometry(*ws.analysis_alnsp_button_geometry)
        self.set_range_button.setGeometry(*ws.set_range_geometry)
        self.max_altitude_label.setGeometry(*ws.max_altitude_label_geometry)
        self.max_speed_label.setGeometry(*ws.max_speed_label_geometry)
        self.max_accel_label.setGeometry(*ws.max_accel_label_geometry)
        self.max_altitude.setGeometry(*ws.max_altitude_geometry)
        self.max_speed.setGeometry(*ws.max_speed_geometry)
        self.max_accel.setGeometry(*ws.max_accel_geometry)
        

        self.max_altitude_label.setFont(ws.font_max_alti_label_text)
        self.max_speed_label.setFont(ws.font_max_speed_label_text)
        self.max_accel_label.setFont(ws.font_max_accel_label_text)
        self.max_altitude.setFont(ws.font_max_alti_text)
        self.max_speed.setFont(ws.font_max_speed_text)
        self.max_accel.setFont(ws.font_max_accel_text)

        self.analysis_button.clicked.connect(self.start_analysis)
        self.analysis_angular_button.clicked.connect(self.start_angularGraph)
        self.analysis_alnsp_button.clicked.connect(self.start_alnspGraph)
        self.set_range_button.clicked.connect(self.reset_range)

        path = abspath(__file__)
        dir_path = dirname(path)

    def initGraph(self):
        self.gr_angle = PlotWidget(self, axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.gr_angleSpeed = PlotWidget(self, axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.gr_accel = PlotWidget(self, axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.gr_altitude = PlotWidget(self, axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.gr_speed = PlotWidget(self, axisItems={'bottom': TimeAxisItem(orientation='bottom')})

        self.gr_angle.setGeometry(*ws.gr_angle_geometry)
        self.gr_angleSpeed.setGeometry(*ws.gr_angleSpeed_geometry)
        self.gr_accel.setGeometry(*ws.gr_accel_geometry)
        self.gr_altitude.setGeometry(*ws.gr_angle_geometry)
        self.gr_speed.setGeometry(*ws.gr_angleSpeed_geometry)

        self.gr_angle.addItem(GridItem())
        self.gr_angleSpeed.addItem(GridItem())
        self.gr_accel.addItem(GridItem())
        self.gr_altitude.addItem(GridItem())
        self.gr_speed.addItem(GridItem())

        self.gr_angle.getPlotItem().getAxis('left').setLabel('Degree')
        self.gr_angleSpeed.getPlotItem().getAxis('left').setLabel('Degree/second')
        self.gr_accel.getPlotItem().getAxis('left').setLabel('g(gravity accel)')
        self.gr_altitude.getPlotItem().getAxis('left').setLabel('Altitude')
        self.gr_speed.getPlotItem().getAxis('left').setLabel('Speed')

        self.gr_angle.getPlotItem().addLegend()
        self.gr_angleSpeed.getPlotItem().addLegend()
        self.gr_accel.getPlotItem().addLegend()
        self.gr_altitude.getPlotItem().addLegend()
        self.gr_speed.getPlotItem().addLegend()

        self.curve_roll = self.gr_angle.plot(pen='r', name = "roll")
        self.curve_pitch = self.gr_angle.plot(pen='g',name = "pitch")
        self.curve_yaw = self.gr_angle.plot(pen='b', name = "yaw")

        self.curve_rollSpeed = self.gr_angleSpeed.plot(pen='r', name = "roll speed")
        self.curve_pitchSpeed = self.gr_angleSpeed.plot(pen='g', name = "pitch speed")
        self.curve_yawSpeed = self.gr_angleSpeed.plot(pen='b', name = "yaw speed")

        self.curve_xaccel = self.gr_accel.plot(pen='r', name = "x acc")
        self.curve_yaccel = self.gr_accel.plot(pen='g',name = "y acc")
        self.curve_zaccel = self.gr_accel.plot(pen='b',name ="z acc")

        self.curve_altitude = self.gr_altitude.plot(pen='g', name = "altitude")

        self.curve_xspeed = self.gr_speed.plot(pen='r', name = "x speed")
        self.curve_yspeed = self.gr_speed.plot(pen='g', name = "y speed")
        self.curve_zspeed = self.gr_speed.plot(pen='b', name = "z speed")

        self.gr_altitude.hide()
        self.gr_speed.hide()
        self.analysis_angular_button.setEnabled(False)
        self.analysis_alnsp_button.setEnabled(True)

    def start_angularGraph(self):
        self.gr_altitude.hide()
        self.gr_speed.hide()
        self.gr_angle.show()
        self.gr_angleSpeed.show()
        self.gr_accel.show()
        self.analysis_angular_button.setEnabled(False)
        self.analysis_alnsp_button.setEnabled(True)

        
    def start_alnspGraph(self):
            self.gr_angle.hide()
            self.gr_angleSpeed.hide()
            self.gr_accel.hide()
            self.gr_altitude.show()
            self.gr_speed.show()
            self.analysis_angular_button.setEnabled(True)
            self.analysis_alnsp_button.setEnabled(False)

    def initMenubar(self):
        self.statusBar()

        change_Action = QAction('Real-Time Viewer', self)
        change_Action.setShortcut('Ctrl+L')
        change_Action.triggered.connect(self.gomain)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        # 스타일시트 수정
        self.setStyleSheet("""
            QMenuBar {
                background-color: rgb(50,50,50);
                color: rgb(255,255,255);
                border: 1px solid rgb(50,50,50);
            }
            QMenu {
                background-color: rgb(255,255,255);  /* 메뉴 드롭다운 창의 배경색 */
                color: rgb(0,0,0);          /* 메뉴 항목의 글씨 색 */
                border: 1px solid rgb(50,50,50);  /* 메뉴 드롭다운 창의 테두리 색 */
            }
            QMenu::item::selected {
                background-color: rgb(50,50,50);  /* 선택된 메뉴 항목의 배경색 */
                color: rgb(255,255,255);          /* 선택된 메뉴 항목의 글씨 색 */
            }
        """)

        filemenu = menubar.addMenu('Menu')
        filemenu.addAction(change_Action)

    def start_analysis(self):
        self.csv_name = self.csv_name_edit.text()
        # Combine the log directory and the CSV file name to create the full file path
        csv_path = os.path.join(self.log_dir, self.csv_name)
        
        try:
            # Use the complete path for reading the CSV file
            alldata = pd.read_csv(csv_path).to_numpy()
            
            # Rest of your data processing code
            init_time = alldata[0,0]*3600 + alldata[0,1]*60 + alldata[0,2] + alldata[0,3]*0.01
            self.timespace = alldata[:,0]*3600000 + alldata[:,1]*60000 + alldata[:,2]*1000 + alldata[:,3]*10

            # Other data processing code...

            self.start_angularGraph()

        except Exception as e:
            print(f"Error: {e}")  # Print the error for debugging
            msg_box = QMessageBox(self)
            self.setStyleSheet("""
                QMessageBox {
                    background-color: black;
                    color: white;
                }
                QLabel {
                    background-color: black;           
                    color: white;
                }
                QPushButton {
                    background-color: white;
                    color: black;
                }
                QInputDialog {
                    background-color: black;
                    color: black;
                }
                QLineEdit {
                    background-color: white;
                    color: black;
                }
            """)
            msg_box.warning(self, "warning", "File open error")
            

    def gomain(self):
        self.goto("main")

    def reset_range(self):
        if self.timespace is not None:
            self.gr_angle.setXRange(min(self.timespace),max(self.timespace))
            self.gr_angleSpeed.setXRange(min(self.timespace),max(self.timespace))
            self.gr_accel.setXRange(min(self.timespace),max(self.timespace))
            self.gr_speed.setXRange(min(self.timespace),max(self.timespace))
            self.gr_altitude.setXRange(min(self.timespace),max(self.timespace))

class window(QMainWindow):
    def __init__(self,datahub):
        self.app = QApplication(argv)
        super().__init__()
        self.datahub = datahub

        self.initUI()
        self.initWindows()
        self.goto("main")

    def initUI(self):
        self.resize(*ws.full_size)
        self.setWindowTitle('I-link')
        self.setStyleSheet(ws.mainwindow_color) 

        path = abspath(__file__)
        dir_path = dirname(path)
        file_path = join(dir_path, 'window_logo.ico')
        self.setWindowIcon(QIcon(file_path))


    def initWindows(self):
        self.mainwindow = MainWindow(self.datahub)
        self.subwindow = SubWindow(self.datahub)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        self.stacked_widget.addWidget(self.mainwindow)
        self.stacked_widget.addWidget(self.subwindow)

        self.mainwindow.gotoSignal.connect(self.goto)
        self.subwindow.gotoSignal.connect(self.goto)

    @pyqtSlot(str)
    def goto(self, name):
        if name == "main":
            self.stacked_widget.setCurrentWidget(self.mainwindow)

        if name == "sub":
            self.stacked_widget.setCurrentWidget(self.subwindow)
            self.subwindow.csv_name_edit.setText("{}".format(self.datahub.file_Name))

    def start(self):
        self.show()
        
    def setEventLoop(self):
        exit(self.app.exec_())

    def closeEvent(self, event):
        self.datahub.communication_stop()
        self.datahub.datasaver_stop()
        try:
            self.mainwindow.enuviewer.stop()
        except Exception:
            pass
        event.accept()