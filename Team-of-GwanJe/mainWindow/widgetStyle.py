from pyautogui import size as max_window_size
from PyQt5.QtGui import QFont
import numpy as np
import sys, os

width, height = max_window_size()
height = int(0.95*height)

a = width
b = height
# a : 가로, b : 세로
full_size = np.array([a,b]).astype(int)

btn_x0   = a * 0.72     # 패널 좌상단 x
btn_y0   = b * 0.42     # 패널 좌상단 y
btn_w    = a * 0.085    # 버튼 가로
btn_h    = b * 0.050    # 버튼 세로
btn_hgap = a * 0.010    # 가로 간격
btn_vgap = b * 0.015    # 세로 간격

# row-major (윗줄→아랫줄, 각 줄 3개)
buttons_geometry = np.array([
    [btn_x0 + c*(btn_w+btn_hgap),  btn_y0 + r*(btn_h+btn_vgap),  btn_w, btn_h]
    for r in range(3) for c in range(3)
]).astype(int)

mainwindow_color = "background-color: rgb(0,0,0);"
webEngine_geometry = np.array([a*0.001,  b*0.001,  b*0.0001, b*0.0001]).astype(int)

# graph geometry
pw_speed_geometry = np.array([a*0.05,  b*0.17,  a*0.25,  b*0.23]).astype(int)
pw_angleSpeed_geometry = np.array([a*0.05,  b*0.435,  a*0.25,  b*0.23]).astype(int)
pw_accel_geometry = np.array([a*0.05,  b*0.71,  a*0.25,  b*0.23]).astype(int)
pw_altitude_geometry = np.array([a*0.74, b*0.45, a*0.25, b*0.24]).astype(int)
pw_trajectory_geometry = np.array([a*0.74, b*0.23, a*0.25, b*0.18]).astype(int)

speed_title_geometry = np.array([a*0.1, b*0.145, 170, 36]).astype(int)
angleSpeed_title_geometry = np.array([a*0.1, b*0.41, 245, 36]).astype(int)
accel_title_geometry = np.array([a*0.1, b*0.68, 245, 40]).astype(int)
altitude_title_geometry = np.array([a*0.85, b*0.425, 245, 37]).astype(int)
trajectory_title_geometry = np.array([a*0.85, b*0.145, 245, 37]).astype(int)

# checker geometry
vx_checker_geometry = np.array([a*0.01,  b*0.21,  100,  50]).astype(int)
vy_checker_geometry = np.array([a*0.01, b*0.25,  100,  50]).astype(int)
vz_checker_geometry = np.array([a*0.01,   b*0.29,  100,  50]).astype(int)

rollS_checker_geomoetry = np.array([a*0.01,  b*0.485, 100,  50]).astype(int)
pitchS_checker_geomoetry = np.array([a*0.01,  b*0.525, 100,  50]).astype(int)
yawS_checker_geomoetry = np.array([a*0.01,  b*0.565, 100,  50]).astype(int)

ax_checker_geomoetry = np.array([a*0.01,  b*0.765,  100,  30]).astype(int)
ay_checker_geomoetry = np.array([a*0.01,  b*0.805,  100,  30]).astype(int)
az_checker_geomoetry = np.array([a*0.01,  b*0.845,  100,  30]).astype(int)

# gps map
map_geometry = np.array([a*0.79,  b*0.65,  a*0.20,  a*0.16]).astype(int)

# serial port editer geometry
port_text_geometry = np.array([a*0.575,  b*0.865, a*0.04, a*0.01125]).astype(int)
port_edit_geometry = np.array([a*0.625,  b*0.865,  a*0.04, a*0.01125]).astype(int)
sender_port_text_geometry = np.array([a*0.68,  b*0.865, a*0.04, a*0.01125]).astype(int)
sender_port_edit_geometry = np.array([a*0.73,  b*0.865,  a*0.04, a*0.01125]).astype(int)

# serial baudrate editer geometry
baudrate_text_geometry = np.array([a*0.575,  b*0.905, a*0.05, a*0.01125]).astype(int)
baudrate_edit_geometry = np.array([a*0.625,  b*0.905,  a*0.04, a*0.01125]).astype(int)
sender_baudrate_text_geometry = np.array([a*0.68,  b*0.905, a*0.05, a*0.01125]).astype(int)
sender_baudrate_edit_geometry = np.array([a*0.73,  b*0.905,  a*0.04, a*0.01125]).astype(int)

# start/stop button geometry
start_geometry = np.array([a*0.41, b*0.85, 0.07*a,  0.05*a ]).astype(int)
stop_geometry = np.array([a*0.495, b*0.85, 0.07*a, 0.05*a]).astype(int)
reset_geometry = np.array([a*0.325, b*0.85, 0.07*a, 0.05*a]).astype(int)
status_geometry = np.array([a*0.67, b*0.79, 0.1*a, 0.05*a]).astype(int)

# buttons geometry
launch1_geometry = np.array([a*0.77, b*0.54, 0.035*a,  0.035*a ]).astype(int)
launch2_geometry = np.array([a*0.77, b*0.44, 0.035*a,  0.035*a ]).astype(int)

launch_stop_geometry = np.array([a*0.95, b*0.54, 0.035*a,  0.035*a ]).astype(int)
emergency_staging_geometry = np.array([a*0.95, b*0.44, 0.035*a,  0.035*a ]).astype(int)

emergency_parachute_geometry = np.array([a*0.84, b*0.44, 0.035*a,  0.035*a ]).astype(int)
staging_stop_geometry = np.array([a*0.88, b*0.44, 0.035*a,  0.035*a ]).astype(int)
nc1_geometry = np.array([a*0.8245, b*0.54, 0.035*a,  0.035*a ]).astype(int)
nc2_geometry = np.array([a*0.86, b*0.54, 0.035*a,  0.035*a ]).astype(int)
nc3_geometry = np.array([a*0.8955, b*0.54, 0.035*a,  0.035*a ]).astype(int)


# rocket animation
model_geometry = np.array([a*0.30,  b*0.14,  a*0.44, a*0.25]).astype(int)
speed_label_geometry = np.array([a*0.33, b*0.57, 600, 45]).astype(int)
altitude_label_geometry = np.array([a*0.53, b*0.57, 600, 45]).astype(int)
roll_label_geometry = np.array([a*0.33, b*0.64, 340, 40]).astype(int)
pitch_label_geometry = np.array([a*0.33, b*0.71, 340, 40]).astype(int)
yaw_label_geometry = np.array([a*0.33, b*0.78, 340, 40]).astype(int)
rollspeed_label_geometry = np.array([a*0.45, b*0.64, 510, 45]).astype(int)
pitchspeed_label_geometry = np.array([a*0.45, b*0.71, 510, 45]).astype(int)
yawspeed_label_geometry = np.array([a*0.45, b*0.78, 510, 45]).astype(int)
xacc_label_geometry = np.array([a*0.63, b*0.64, 335, 40]).astype(int)
yacc_label_geometry = np.array([a*0.63, b*0.71, 335, 40]).astype(int)
zacc_label_geometry = np.array([a*0.63, b*0.78, 335, 40]).astype(int)

# 
cmd_geometry = np.array([a*0.001, 0.001*a,  a*0.0001,  0.0001*a]).astype(int)

# 상단 로고들
team_logo_geometry = np.array([a*0.23, 0.022*b,  2200,  200]).astype(int)
irri_logo_geometry = np.array([a*0.797, -0.075*b,  650,  500]).astype(int)
patch22_logo_geometry = np.array([a*0.01, 0.0182*b,  155, 130]).astype(int)
patch24_logo_geometry = np.array([a*0.06, 0.024*b,  150,  115]).astype(int)
patch_logo_geometry = np.array([a*0.16, 0.02*b,  200,  200]).astype(int)

# all fonts
font_portText = QFont()
font_portText.setPointSize(11)

font_baudrate = QFont()
font_baudrate.setPointSize(11)

checker_font = QFont()
checker_font.setPointSize(10)

font_guideText = QFont()
font_guideText.setPointSize(10)

font_trajectory_title = QFont()
font_trajectory_title.setPointSize(13)

font_angle_title = QFont()
font_angle_title.setPointSize(13)

font_angleSpeed_title = QFont()
font_angleSpeed_title.setPointSize(13)

font_accel_title = QFont()
font_accel_title.setPointSize(13)

font_altitude_title = QFont()
font_altitude_title.setPointSize(13)

font_speed_title = QFont()
font_speed_title.setPointSize(13)

font_start_text = QFont()
font_start_text.setPointSize(20)

font_stop_text = QFont()
font_stop_text.setPointSize(20)

font_reset_text = QFont()
font_reset_text.setPointSize(20)

font_button_text = QFont()
font_button_text.setPointSize(7)

font_status_text = QFont()
font_status_text.setFamily("VCR OSD Mono")
font_status_text.setPointSize(12)
# next to rocket animation
font_speed_text = QFont()
font_speed_text.setFamily("VCR OSD Mono")
font_speed_text.setPointSize(20)

font_altitude_text = QFont()
font_altitude_text.setFamily("VCR OSD Mono")
font_altitude_text.setPointSize(20)

font_roll_text = QFont()
font_roll_text.setFamily("VCR OSD Mono")
font_roll_text.setPointSize(12)

font_pitch_text = QFont()
font_pitch_text.setFamily("VCR OSD Mono")
font_pitch_text.setPointSize(12)

font_yaw_text = QFont()
font_yaw_text.setFamily("VCR OSD Mono")
font_yaw_text.setPointSize(12)

font_rollspeed_text = QFont()
font_rollspeed_text.setFamily("VCR OSD Mono")
font_rollspeed_text.setPointSize(12)

font_pitchspeed_text = QFont()
font_pitchspeed_text.setFamily("VCR OSD Mono")
font_pitchspeed_text.setPointSize(12)

font_yawspeed_text = QFont()
font_yawspeed_text.setFamily("VCR OSD Mono")
font_yawspeed_text.setPointSize(12)

font_xacc_text = QFont()
font_xacc_text.setFamily("VCR OSD Mono")
font_xacc_text.setPointSize(12)

font_yacc_text = QFont()
font_yacc_text.setFamily("VCR OSD Mono")
font_yacc_text.setPointSize(12)

font_zacc_text = QFont()
font_zacc_text.setFamily("VCR OSD Mono")
font_zacc_text.setPointSize(12)


start_status = 'Program start.'
stop_status = 'Program stop.'
wait_status = 'Wait for start'

guide = """
MISSION CONTROL CENTER
"""

### Sub window ###

csv_name_geometry = np.array([a*0.8, 0.8*b,  a*0.1, a*0.02]).astype(int)
analysis_button_geometry = np.array([a*0.8, 0.85*b,  a*0.1, a*0.02]).astype(int)
analysis_angular_button_geometry = np.array([a*0.73, 0.7*b, a*0.10, a*0.02]).astype(int)
analysis_alnsp_button_geometry = np.array([a*0.87, 0.7*b, a*0.10, a*0.02]).astype(int)
set_range_geometry = np.array([a*0.8, 0.55*b,  a*0.1, a*0.02]).astype(int)

gr_angle_geometry = np.array([a*0.1,  b*0.04,  a*0.6,  b*0.28]).astype(int)
gr_angleSpeed_geometry = np.array([a*0.1,  b*0.36,  a*0.6,  b*0.28]).astype(int)
gr_accel_geometry = np.array([a*0.1,  b*0.68,  a*0.6,  b*0.28]).astype(int)

max_altitude_label_geometry = np.array([a*0.75,  b*0.15,  320, 50]).astype(int)
max_speed_label_geometry = np.array([a*0.75,  b*0.15+50,  300, 50]).astype(int)
max_accel_label_geometry = np.array([a*0.75,  b*0.15+100,  300, 50]).astype(int)

max_altitude_geometry = np.array([a*0.75+400,  b*0.15,  300, 50]).astype(int)
max_speed_geometry = np.array([a*0.75+400,  b*0.15+50,  300, 50]).astype(int)
max_accel_geometry = np.array([a*0.75+400,  b*0.15+100,  300, 50]).astype(int)




font_max_alti_label_text = QFont()
font_max_alti_label_text.setFamily("VCR OSD Mono")
font_max_alti_label_text.setPointSize(15)

font_max_speed_label_text = QFont()
font_max_speed_label_text.setFamily("VCR OSD Mono")
font_max_speed_label_text.setPointSize(15)

font_max_accel_label_text = QFont()
font_max_accel_label_text.setFamily("VCR OSD Mono")
font_max_accel_label_text.setPointSize(15)

font_max_alti_text = QFont()
font_max_alti_text.setFamily("VCR OSD Mono")
font_max_alti_text.setPointSize(15)

font_max_speed_text = QFont()
font_max_speed_text.setFamily("VCR OSD Mono")
font_max_speed_text.setPointSize(15)

font_max_accel_text = QFont()
font_max_accel_text.setFamily("VCR OSD Mono")
font_max_accel_text.setPointSize(15)