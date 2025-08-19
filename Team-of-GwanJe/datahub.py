from numpy import empty, append
import numpy as np

class Datahub:

    def __init__ (self):
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

        self.Easts       = empty(0)
        self.Norths      = empty(0)
        self.Ups         = empty(0)
        
        self.vE          = empty(0)
        self.vN          = empty(0)
        self.vU          = empty(0)

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
        
        #map view trigger
        self.trigger_python = 0
            
    def communication_start(self):
        self.iscommunication_start=True
        
    def communication_stop(self):
        self.iscommunication_start=False
    
    def check_communication_error(self):
        while True:
            if self.serial_port_error==0 or self.serial_port_error==1:
                return self.serial_port_error
    
    def datasaver_start(self):
        self.isdatasaver_start=1

    def datasaver_stop(self):
        self.isdatasaver_start=0
    
    @staticmethod
    def _quat_to_euler_deg(q0, q1, q2, q3):
        # scalar-first (w, x, y, z), ZYX (yaw-pitch-roll)
        sinr_cosp = 2*(q0*q1 + q2*q3)
        cosr_cosp = 1 - 2*(q1*q1 + q2*q2)
        roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

        sinp = 2*(q0*q2 - q3*q1)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.degrees(np.arcsin(sinp))

        siny_cosp = 2*(q0*q3 + q1*q2)
        cosy_cosp = 1 - 2*(q2*q2 + q3*q3)
        yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

        return roll, pitch, yaw
    
    def update(self,datas):
        """Update Datas received from rocket"""
        self.hours       = append(self.hours,datas[0])
        self.mins        = append(self.hours,datas[1])
        self.secs        = append(self.hours,datas[2])
        self.tenmilis    = append(self.hours,datas[3])

        self.Easts       = append(self.hours,datas[4])
        self.Norths      = append(self.hours,datas[5])
        self.Ups         = append(self.hours,datas[6])
        
        self.vE          = append(self.hours,datas[7])
        self.vN          = append(self.hours,datas[8])
        self.vU          = append(self.hours,datas[9])

        self.speed       = append(self.hours,datas[10])
        self.yspeed      = append(self.hours,datas[11])
        self.zspeed      = append(self.hours,datas[12])

        self.Xaccels     = append(self.hours,datas[13])
        self.Yaccels     = append(self.hours,datas[14])
        self.Zaccels     = append(self.hours,datas[15])

        self.q0          = append(self.hours,datas[16])
        self.q1          = append(self.hours,datas[17])
        self.q2          = append(self.hours,datas[18])
        self.q3          = append(self.hours,datas[19])

        self.rollSpeeds  = append(self.hours,datas[20])
        self.pitchSpeeds = append(self.hours,datas[21])
        self.yawSpeeds   = append(self.hours,datas[22])


        self.r, self.p, self.y = self._quat_to_euler_deg(self.q0, self.q1, self.q2, self.q3)

        self.rolls       = append(self.rolls, self.r)
        self.pitchs      = append(self.pitchs, self.p)
        self.yaws        = append(self.yaws, self.y)

    def update_from_row(self, row):
        (hours, mins, secs, tenmilis, E, N, U, v_E, v_N, v_U, a_p, a_y, a_r, q_0, q_1, q_2, q_3, w_p, w_y, w_r) = row

        # time
        self.hours    = append(self.hours, hours)
        self.mins     = append(self.mins, mins)
        self.secs     = append(self.secs, secs)
        self.tenmilis = append(self.tenmilis, tenmilis)

        # ENU & velocity
        self.Easts  = append(self.Easts, E)
        self.Norths = append(self.Norths, N)
        self.Ups    = append(self.Ups, U)

        self.vE = append(self.vE, v_E)
        self.vN = append(self.vN, v_N)
        self.vU = append(self.vU, v_U)

        # accel
        self.Xaccels = append(self.Xaccels, a_p)
        self.Yaccels = append(self.Yaccels, a_y)
        self.Zaccels = append(self.Zaccels, a_r)

        # gyro
        self.rollSpeeds  = append(self.rollSpeeds,  w_p)
        self.pitchSpeeds = append(self.pitchSpeeds, w_y)
        self.yawSpeeds   = append(self.yawSpeeds,   w_r)

        # quat
        self.q0 = append(self.q0, q_0)
        self.q1 = append(self.q1, q_1)
        self.q2 = append(self.q2, q_2)
        self.q3 = append(self.q3, q_3)

        # euler
        r, p, y = self._quat_to_euler_deg(q_0, q_1, q_2, q_3)
        self.rolls  = append(self.rolls,  r)
        self.pitchs = append(self.pitchs, p)
        self.yaws   = append(self.yaws,   y)

    def clear(self):
        self.hours       = empty(0)
        self.mins        = empty(0)
        self.secs        = empty(0)
        self.tenmilis    = empty(0)

        self.Easts       = empty(0)
        self.Norths      = empty(0)
        self.Ups         = empty(0)
        
        self.vE          = empty(0)
        self.vN          = empty(0)
        self.vU          = empty(0)

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

        self.__init__()