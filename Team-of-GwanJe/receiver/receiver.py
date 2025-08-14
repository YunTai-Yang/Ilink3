from struct import unpack
from serial import Serial, PARITY_NONE, STOPBITS_TWO, EIGHTBITS
from numpy import sum, around
from threading import Thread
from time import sleep
#from datahub import Datahub

import numpy as np


##'A', 'B', len, hours, mins, secs, tenmilis, E, N, U, v_E, v_N, v_U, a_p, a_y, a_r, w_p, w_y, w_r, q_0, q_1, q_2, q_3, checksum, 'Z'
##'A', 'B', len, order, checksum, 'Z' 

class Receiver(Thread):
    def __init__(self, datahub):
        super().__init__( daemon=True )
        self.datahub = datahub
        self.first_time = True
        self.ser = None


    def setSerial(self,myport,mybaudrate):
            self.ser = Serial(port=myport,
                                    baudrate = mybaudrate,
                                    parity=PARITY_NONE,
                                    stopbits=STOPBITS_TWO,
                                    bytesize=EIGHTBITS,
                                    timeout=0.1)

    def _decode_data(self, data_bytes):
        decode_data = unpack('<18f', data_bytes)
        print(decode_data)

        if sum(decode_data[4:-1])-decode_data[-1]<1:
            all_data = around(decode_data,4)
            if len(all_data)>=1:
                self.datahub.update(all_data)


    def run(self):
        
        hdrf = 0
        idxn = 0
        
        msgd = np.zeros( 74, dtype=np.uint8 )
        
        while True:
            try:
                if self.datahub.iscommunication_start:
                    if self.first_time:
                        self.setSerial( self.datahub.mySerialPort, self.datahub.myBaudrate )
                        self.first_time=False

                    if not self.ser.is_open:
                        self.ser.open()
                        print( "opened" )

                    self.datahub.serial_port_error=0
                    byte = self.ser.read()
                    
                    if ( hdrf == 2 ):
                        msgd[idxn] = np.frombuffer( byte, np.uint8 )
                        
                        idxn += 1

                        if ( idxn == 74 ):
                            data = np.frombuffer( msgd[2:], np.float32 )
                            
                            if ( np.sum( data[4:-1] ) == data[-1] ):
                                # print( 'good' )
                                
                                self.datahub.update(data)
                                
                            hdrf = 0
                            idxn = 0
                         
                    elif ( hdrf == 0 ):   
                        if ( byte == b"A" ):
                            msgd[idxn] = np.frombuffer( byte, np.uint8 )
    
                            hdrf += 1
                            idxn += 1
                            
                        else:
                            hdrf = 0
                            idxn = 0
                            
                    elif ( hdrf == 1 ):
                        if ( byte == b"B" ):
                            msgd[idxn] = np.frombuffer( byte, np.uint8 )
    
                            hdrf += 1
                            idxn += 1
                            
                        else:
                            hdrf = 0
                            idxn = 0

                    else:
                        hdrf = 0
                        idxn = 0

                else:
                    self.datahub.communication_start()

                    if self.ser != None and self.ser.is_open :
                        self.ser.close()
                    sleep(0.05)
            except:
                self.datahub.serial_port_error=1


if __name__=="__main__":
    receiver = Receiver()
    receiver.run()