from threading import Thread, Lock

from datasaver.datasaver import DataSaver
from communication.receiver import Receiver
from communication.sender   import Sender
from mainWindow.mainWindow  import window
from datahub import Datahub


class Thread_Receiver(Thread):
    """Receiver(Thread)를 백그라운드에서 돌리는 래퍼 (기존 구조 유지)."""
    def __init__(self, datahub):
        super().__init__(daemon=True)
        self.datahub  = datahub
        self.receiver = Receiver(self.datahub)  # Receiver 자체가 Thread

    def run(self):
        self.receiver.start()  # 실제 수신 스레드 시작


class Thread_DataSaver(Thread):
    """DataSaver를 백그라운드에서 돌리는 래퍼 (기존 구조 유지)."""
    def __init__(self, datahub):
        super().__init__(daemon=True)
        self.datahub  = datahub
        self.datasaver = DataSaver(self.datahub)

    def run(self):
        self.datasaver.start()


class Master:
    def __init__(self):
        self.datahub = Datahub()

        # UI
        self.mainWindow = window(self.datahub)

        # 백그라운드 스레드
        self.rx_thread  = Thread_Receiver(self.datahub)
        self.ds_thread  = Thread_DataSaver(self.datahub)

        # 시리얼 write 보호용(여러 곳에서 write할 수 있으므로)
        self.serial_lock = Lock()

        # Sender: ★ Receiver가 연 포트를 '공유'해서 사용하도록 구성
        #  - get_serial: 현재 열려있는 Serial 핸들 반환
        #  - serial_lock: write 동시성 보호
        self.sender = Sender(
            self.datahub,
            get_serial=lambda: self.rx_thread.receiver.ser,  # 공유!
            serial_lock=self.serial_lock
        )
        self.sender.daemon = True

        # UI가 버튼 토글 등에서 바로 보낼 수 있도록 참조 넘김(옵션)
        try:
            self.mainWindow.sender = self.sender
        except Exception:
            pass

    def run(self):
        # 순서: 수신기 먼저 → (선택) 데이터세이버 → Sender
        self.rx_thread.start()
        self.ds_thread.start()
        self.sender.start()

        # UI 루프 시작 (여기서 블록)
        self.mainWindow.start()
        self.mainWindow.setEventLoop()

if __name__ == "__main__":
    master = Master()
    # 전체화면으로 띄우고 시작
    master.mainWindow.showMaximized()
    master.run()