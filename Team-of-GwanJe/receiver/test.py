from threading import Thread

from serial import Serial

import numpy as np

from struct import unpack



class Receiver( Thread ):
    
    def __init__( self, serial: Serial, _len, n=10000 ):
        self.data = np.zeros((n,_len))
        
        self.serial = serial
        self._len   = _len
        self.alive  = True
        
        super().__init__( daemon=True )
    
    
    def run( self ):
        hdrf = 0
        idxn = 0
        
        serial = self.serial
        _len   = self._len
        
        msgd = np.zeros( _len, dtype=np.uint8 )

        while self.alive:
            byte = serial.read()
            
            if ( hdrf == 2 ):
                msgd[idxn] = np.frombuffer( byte, np.uint8 )
            
                idxn += 1
                
                if ( idxn == _len ):
                    hdrf = 0
                    idxn = 0

                    print( msgd )
                    
                    print( np.frombuffer( msgd[2:_len], dtype=np.float32 ) )
                    
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
    
    
    def join( self ):
        self.alive = False
        
        print( "I'm dead" )
        
        
if __name__ == "__main__":
    serial = Serial( "COM7", baudrate=9600 )
    
    receiver = Receiver( serial, 74 )
    
    receiver.start()
    
    input( 'suspend' )
    
    receiver.join()
    