from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsDropShadowEffect , QPushButton,QLineEdit, QLabel, QMessageBox, QInputDialog, QCheckBox, QStackedWidget, QAction, QFrame, QWidget, QVBoxLayout
from pathlib import Path
from PyQt5.QtCore import QThread, QUrl, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QIcon, QPixmap
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
#################
from numpy import zeros, array, cross, reshape, sin, cos, deg2rad, rad2deg
from numpy.random import rand
from numpy.linalg import norm

from matplotlib.pyplot import figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt

from datetime import timedelta
from os.path import abspath, dirname, join, exists
from sys import exit, argv

from pandas import read_csv

from . import widgetStyle as ws

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
        if len(self.datahub.speed) == 0:
            pass

        else:
            if len(self.datahub.speed) <= self.x_ran :
                n = len(self.datahub.speed) 
                self.rollSpeed[-n:] = self.datahub.rollSpeeds[-n:]
                self.pitchSpeed[-n:] = self.datahub.pitchSpeeds[-n:]
                self.yawSpeed[-n:] = self.datahub.yawSpeeds[-n:]
                self.xaccel[-n:] = self.datahub.Xaccels[-n:]
                self.yaccel[-n:] = self.datahub.Yaccels[-n:]
                self.zaccel[-n:] = self.datahub.Zaccels[-n:]
                self.xspeed[-n:] = self.datahub.speed[-n:]
                self.yspeed[-n:] = self.datahub.yspeed[-n:]
                self.zspeed[-n:] = self.datahub.zspeed[-n:]
                hours = self.datahub.hours[-n:] * 3600
                minutes = self.datahub.mins[-n:] * 60
                miliseconds = self.datahub.tenmilis[-n:] * 0.01
                seconds = self.datahub.secs[-n:]
                totaltime = hours + minutes + miliseconds + seconds
                self.starttime = self.datahub.hours[0]*3600 + self.datahub.mins[0]*60 + self.datahub.tenmilis[0]*0.01+ self.datahub.secs[0]
                self.time[-n:] = totaltime - self.starttime
            
            else : 
                self.rollSpeed[:] = self.datahub.rollSpeeds[-self.x_ran:]
                self.pitchSpeed[:] = self.datahub.pitchSpeeds[-self.x_ran:]
                self.yawSpeed[:] = self.datahub.yawSpeeds[-self.x_ran:]
                self.xaccel[:] = self.datahub.Xaccels[-self.x_ran:]
                self.yaccel[:] = self.datahub.Yaccels[-self.x_ran:]
                self.zaccel[:] = self.datahub.Zaccels[-self.x_ran:]
                self.xspeed[:] = self.datahub.speed[-self.x_ran:]
                self.yspeed[:] = self.datahub.yspeed[-self.x_ran:]
                self.zspeed[:] = self.datahub.zspeed[-self.x_ran:]               
                hours = self.datahub.hours[-self.x_ran:] * 3600
                minutes = self.datahub.mins[-self.x_ran:] * 60
                miliseconds = self.datahub.tenmilis[-self.x_ran:] * 0.01
                seconds = self.datahub.secs[-self.x_ran:]
                totaltime = hours + minutes + miliseconds + seconds
                self.time[:] = totaltime - self.starttime

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


class MapViewer_Thread(QThread):
    def __init__(self, mainwindow, datahub):
        super().__init__()
        self.mainwindow = mainwindow
        self.datahub = datahub

        # 1) 이 파일(mainWindow.py)이 있는 디렉터리
        base_dir = Path(__file__).resolve().parent

        # 2) map.html 파일 경로
        map_path = base_dir / "map.html"
        if not map_path.exists():
            raise FileNotFoundError(f"map.html을 찾을 수 없습니다: {map_path}")

        # 3) HTML 읽어서 사이즈 조정
        html = map_path.read_text(encoding="utf-8")
        new_width  = f"{ws.map_geometry[2]}px"
        new_height = f"{ws.map_geometry[3]}px"
        html = html.replace("width: 576px;",  f"width: {new_width};")
        html = html.replace("height: 345px;", f"height: {new_height};")
        map_path.write_text(html, encoding="utf-8")

        # 4) QWebEngineView에 로드
        self.view = QWebEngineView(self.mainwindow.container)
        self.view.setGeometry(*ws.map_geometry)
        self.view.load(QUrl.fromLocalFile(str(map_path)))
        self.view.show()

    def on_load_finished(self):
        # Get the QWebEnginePage object
        page = self.view.page()
        # Inject a JavaScript function to update the marker's location
        self.script = f"""
        var lat = 37.45162;
        var lng = 126.65058;
        var map = L.map("map").setView([lat,lng], 17);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
            maxZoom: 18,
        }}).addTo(map);
        var marker = L.marker([lat,lng]).addTo(map);
        /*
        trigger is a variable which update a map view according to their location
        */
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

        # Create a QTimer to call the updateMarker function every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_marker)
        self.timer.start(1000)

    def update_marker(self):
        # Wait for receiving data...
        if len(self.datahub.latitudes) == 0:
            return
        # Call the JavaScript function to update the marker's location
        lat = self.datahub.latitudes[-1]
        lng = self.datahub.longitudes[-1]
        self.view.page().runJavaScript(f"updateMarker({lat}, {lng}, {len(self.datahub.latitudes)})")

    # Connect the QWebEngineView's loadFinished signal to the on_load_finished slot
    def run(self):
        self.view.loadFinished.connect(self.on_load_finished)

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

    def quaternion_from_euler(self, roll, pitch, yaw):
        # Roll (X축 회전), Pitch (Y축 회전), Yaw (Z축 회전)
        roll = deg2rad(roll) / 2
        pitch = deg2rad(pitch) / 2
        yaw = deg2rad(yaw) / 2

        cy = cos(yaw)
        sy = sin(yaw)
        cp = cos(pitch)
        sp = sin(pitch)
        cr = cos(roll)
        sr = sin(roll)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return array([qw, qx, qy, qz])

    def euler_from_quaternion(self, quat):
        qw, qx, qy, qz = quat

        # 오일러 각도를 계산
        roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = np.arcsin(2.0 * (qw * qy - qz * qx))
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # 라디안을 도로 변환
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)

        return roll, pitch, yaw

    def update_pose(self):
        if len(self.datahub.speed) == 0:
            return

        # 오일러 각도에서 쿼터니언을 생성
        quat = self.quaternion_from_euler(self.datahub.rolls[-1], self.datahub.pitchs[-1], self.datahub.yaws[-1])

        # 로켓의 변환 초기화
        self.rocket_mesh.resetTransform()

        # 쿼터니언에서 직접 회전 적용
        roll, pitch, yaw = self.euler_from_quaternion(quat)
        self.rocket_mesh.rotate(pitch, 1, 0, 0)  # X축을 기준으로 Pitch 적용
        self.rocket_mesh.rotate(roll, 0, 1, 0)  # Y축을 기준으로 Roll 적용
        self.rocket_mesh.rotate(yaw, 0, 0, 1)  # Z축을 기준으로 Yaw 적용




        # 데이터에 따라 UI 레이블 업데이트
        self.speed_label.setText(f'Speed {self.datahub.speed[-1]:.2f}m/s')
        self.altitude_label.setText(f'Altitude {self.datahub.altitude[-1]:.2f}m')
        self.roll_label.setText(f'Roll : {roll:.2f}°')
        self.pitch_label.setText(f'Pitch : {pitch:.2f}°')
        self.yaw_label.setText(f'Yaw : {yaw:.2f}°')
        self.rollspeed_label.setText(f'Roll_speed : {self.datahub.rollSpeeds[-1]:.2f}°/s')
        self.pitchspeed_label.setText(f'Pitch_speed : {self.datahub.pitchSpeeds[-1]:.2f}°/s')
        self.yawspeed_label.setText(f'Yaw_speed : {self.datahub.yawSpeeds[-1]:.2f}°/s')
        self.xacc_label.setText(f'X_acc : {self.datahub.Xaccels[-1]:.2f}g')
        self.yacc_label.setText(f'Y_acc : {self.datahub.Yaccels[-1]:.2f}g')
        self.zacc_label.setText(f'Z_acc : {self.datahub.Zaccels[-1]:.2f}g')

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

        self.initMenubar()

        self.mapviewer.start()
        self.graphviewer.start()
        self.rocketviewer.start()

        self.resetcheck = 0
        
    def initUI(self):

        """Set Buttons"""
        self.start_button = QPushButton("Start",self.container)
        self.stop_button = QPushButton("Stop",self.container)
        self.reset_button = QPushButton("Reset",self.container)
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
            FileName, ok = input_dialog.getText(self, 'Input Dialog', 'Enter your File Name', QLineEdit.Normal, "Your File Name")
        
            file_dir = dirname(self.dir_path)
            file_path = join(file_dir,FileName)+".csv"
            if exists(file_path):
                msg_box = QMessageBox(self)
                msg_box.setStyleSheet("QMessageBox {background-color: white;}")
                msg_box.information(self,"information","Same file already exist")
            else:
                if ok:
                    self.datahub.mySerialPort=self.rf_port_edit.text()
                    self.datahub.myBaudrate = self.baudrate_edit.text()
                    self.datahub.file_Name = FileName+'.csv'
                    self.datahub.communication_start()

                    print( "changed" )
                    
                    self.datahub.serial_port_error=-1
                    if self.datahub.check_communication_error():
                        warning_msg_box = QMessageBox(self)
                        warning_msg_box.setStyleSheet("QMessageBox {background-color: white;}")
                        warning_msg_box.warning(self,"warning","Check the Port or Baudrate again.")
                        self.datahub.communication_stop()
                    else:
                        self.datahub.datasaver_start()
                        self.now_status.setText(ws.start_status)
                        self.start_button.setEnabled(False)
                        self.stop_button.setEnabled(True)
                        self.rf_port_edit.setEnabled(False)
                        self.baudrate_edit.setEnabled(False)
                        self.shadow_start_button.setOffset(0)
                        self.shadow_stop_button.setOffset(6)
                        self.shadow_reset_button.setOffset(6)
                self.datahub.serial_port_error=-1
        else:
            self.datahub.communication_start()
            self.datahub.serial_port_error=-1
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
        self.datahub.communication_stop()
        self.now_status.setText(ws.stop_status)
        self.now_status.setStyleSheet("color:#00FF00;")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.rf_port_edit.setEnabled(False)
        self.shadow_start_button.setOffset(6)
        self.shadow_stop_button.setOffset(0)
        self.shadow_reset_button.setOffset(6)
        self.resetcheck = 1

    def reset_button_clicked(self):
        request = QMessageBox.question(self,'Message', 'Are you sure to reset?')
        if request == QMessageBox.Yes:
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
        self.graphviewer.curve_xspeed.setVisible(state != Qt.Checked)
    def zspeed_hide_checkbox_state(self,state):
        self.graphviewer.curve_xspeed.setVisible(state != Qt.Checked)        
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
        event.accept()