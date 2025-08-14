from threading import Thread

from datasaver.datasaver import DataSaver
from receiver.receiver   import Receiver
from mainWindow.mainWindow import window
from datahub import Datahub

class Thread_Receiver(Thread):
    def __init__(self,datahub):
        super().__init__()
        self.datahub = datahub

    def run(self):
        receiver = Receiver(self.datahub)
        receiver.start()


class Thread_DataSaver(Thread):
    def __init__(self, datahub):
        super().__init__()
        self.datahub = datahub
        

    def run(self):
        datasaver = DataSaver(self.datahub)
        datasaver.start()


class Master:

    def __init__(self):
        
        self.datahub = Datahub()

        self.mainWindow = window(self.datahub)

        self.datasaver = Thread_DataSaver(self.datahub)

        self.receiver = Receiver(self.datahub)


        self.datasaver.daemon = True
        self.receiver.daemon = True
        
    def run(self):

        self.receiver.start()
        
        self.datasaver.start()

        self.mainWindow.start() 
        
        self.mainWindow.setEventLoop()

if __name__ == "__main__":
    master = Master()

    master.run()
    
    input()