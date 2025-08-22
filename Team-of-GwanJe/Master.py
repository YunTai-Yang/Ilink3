from threading import Thread

from datasaver.datasaver import DataSaver
from communication.receiver   import Receiver
from communication.sender import Sender
from mainWindow.mainWindow import window
from datahub import Datahub


class Thread_Receiver(Thread):
    def __init__(self, datahub):
        super().__init__(daemon=True)
        self.datahub = datahub
        self.receiver = Receiver(self.datahub)

    def run(self):
        self.receiver.start()


class Thread_DataSaver(Thread):
    def __init__(self, datahub):
        super().__init__(daemon=True)
        self.datahub = datahub
        self.datasaver = DataSaver(self.datahub)

    def run(self):
        self.datasaver.start()


class Master:
    def __init__(self):
        self.datahub = Datahub()

        # UI
        self.mainWindow = window(self.datahub)

        # 백그라운드 스레드들
        self.receiver  = Thread_Receiver(self.datahub)
        self.datasaver = Thread_DataSaver(self.datahub)

        # ★ Sender 추가
        self.sender = Sender(self.datahub)
        self.sender.daemon = True

        # (선택) 메인윈도우에서 바로 전송할 수 있게 참조 넘기기
        #   → MainWindow.on_toggle 마지막에 self.sender.enqueue_button(latest) 처럼 사용
        try:
            self.mainWindow.sender = self.sender
        except Exception:
            pass

    def run(self):
        # 백엔드 스레드 시작
        self.receiver.start()
        self.datasaver.start()

        # ★ Sender 시작
        self.sender.start()

        # UI 루프
        self.mainWindow.start()
        self.mainWindow.setEventLoop()


if __name__ == "__main__":
    master = Master()
    master.mainWindow.showMaximized()
    master.run()
    input()
