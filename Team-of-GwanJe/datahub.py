from numpy import empty, append

class Datahub:

    def __init__ (self):
        """
        Communication Status Parameter
        """
        self.iscommunication_start = 0
        self.isdatasaver_start = 0
        self.file_Name = 'Your File Name.csv'
        self.mySerialPort = 'COM6'
        self.myBaudrate = 115200
        self.serial_port_error=-1
        """
        Rocket Status Parameter
        """
        
        self.hours = empty(0)
        self.mins = empty(0)
        self.secs = empty(0)
        self.tenmilis = empty(0)
        self.rolls = empty(0)
        self.pitchs = empty(0)
        self.yaws = empty(0)
        self.rollSpeeds = empty(0)
        self.pitchSpeeds = empty(0)
        self.yawSpeeds = empty(0)
        self.Xaccels = empty(0)
        self.Yaccels = empty(0)
        self.Zaccels = empty(0)
        self.latitudes = empty(0)
        self.longitudes = empty(0)
        self.altitude = empty(0)
        self.speed = empty(0)
        
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
        
    def update(self,datas):
        """Update Datas received from rocket"""
        
        self.hours = append(self.hours,datas[0])
        self.mins = append(self.mins,datas[1])
        self.secs = append(self.secs,datas[2])
        self.tenmilis = append(self.tenmilis,datas[3])
        self.rolls = append(self.rolls,datas[4])
        self.pitchs = append(self.pitchs,datas[5])
        self.yaws = append(self.yaws, datas[6])
        self.rollSpeeds = append(self.rollSpeeds, datas[7])
        self.pitchSpeeds = append(self.pitchSpeeds, datas[8])
        self.yawSpeeds = append(self.yawSpeeds, datas[9])
        self.Xaccels = append(self.Xaccels, datas[10])
        self.Yaccels = append(self.Yaccels, datas[11])
        self.Zaccels = append(self.Zaccels, datas[12])
        self.latitudes = append(self.latitudes, datas[13])
        self.longitudes = append(self.longitudes, datas[14])
        self.altitude = append(self.altitude, datas[15])
        self.speed = append(self.speed, datas[16])


    def clear(self):
        self.hours = empty(0)
        self.mins = empty(0)
        self.secs = empty(0)
        self.tenmilis = empty(0)
        self.rolls = empty(0)
        self.pitchs = empty(0)
        self.yaws = empty(0)
        self.rollSpeeds = empty(0)
        self.pitchSpeeds = empty(0)
        self.yawSpeeds = empty(0)
        self.Xaccels = empty(0)
        self.Yaccels = empty(0)
        self.Zaccels = empty(0)
        self.latitudes = empty(0)
        self.longitudes = empty(0)
        self.altitude = empty(0)
        self.speed = empty(0)

    def __init__ (self):
        """
        Communication Status Parameter
        """
        self.iscommunication_start = 0
        self.isdatasaver_start = 0
        self.file_Name = 'Your File Name.csv'
        self.mySerialPort = 'COM8'
        self.myBaudrate = 115200
        self.serial_port_error=-1
        """
        Rocket Status Parameter
        """
        
        self.hours = empty(0)
        self.mins = empty(0)
        self.secs = empty(0)
        self.tenmilis = empty(0)
        self.rolls = empty(0)
        self.pitchs = empty(0)
        self.yaws = empty(0)
        self.rollSpeeds = empty(0)
        self.pitchSpeeds = empty(0)
        self.yawSpeeds = empty(0)
        self.Xaccels = empty(0)
        self.Yaccels = empty(0)
        self.Zaccels = empty(0)
        self.latitudes = empty(0)
        self.longitudes = empty(0)
        self.altitude = empty(0)
        self.speed = empty(0)
        self.n=0
        
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
        
    def update(self,datas):
        """Update Datas received from rocket"""
        
        self.hours = append(self.hours,datas[0])
        self.mins = append(self.mins,datas[1])
        self.secs = append(self.secs,datas[2])
        self.tenmilis = append(self.tenmilis,datas[3])
        self.rolls = append(self.rolls,datas[4])
        self.pitchs = append(self.pitchs,datas[5])
        self.yaws = append(self.yaws, datas[6])
        self.rollSpeeds = append(self.rollSpeeds, datas[7])
        self.pitchSpeeds = append(self.pitchSpeeds, datas[8])
        self.yawSpeeds = append(self.yawSpeeds, datas[9])
        self.Xaccels = append(self.Xaccels, datas[10])
        self.Yaccels = append(self.Yaccels, datas[11])
        self.Zaccels = append(self.Zaccels, datas[12])
        self.latitudes = append(self.latitudes, datas[13])
        self.longitudes = append(self.longitudes, datas[14])
        self.altitude = append(self.altitude, datas[15])
        self.speed = append(self.speed, datas[16])



    def clear(self):
        self.hours = empty(0)
        self.mins = empty(0)
        self.secs = empty(0)
        self.tenmilis = empty(0)
        self.rolls = empty(0)
        self.pitchs = empty(0)
        self.yaws = empty(0)
        self.rollSpeeds = empty(0)
        self.pitchSpeeds = empty(0)
        self.yawSpeeds = empty(0)
        self.Xaccels = empty(0)
        self.Yaccels = empty(0)
        self.Zaccels = empty(0)
        self.latitudes = empty(0)
        self.longitudes = empty(0)
        self.altitude = empty(0)
        self.speed = empty(0)