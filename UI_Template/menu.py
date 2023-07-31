from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets
from trainer import Trainer
from verify import Verifier
from detect import Detector
from menu_ui import Ui_MainWindow

class Menu:
    
    def __init__(self) -> None:
        # 建構子 : 初始化 UI 介面
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        # Button Event : 將互動元件與程式碼 functions 連接
        self.ui.pushButton_train.clicked.connect(self.pushButton_train_click)
        self.ui.pushButton_verify.clicked.connect(self.pushButton_verify_click)
        self.ui.pushButton_detect.clicked.connect(self.pushButton_detect_click)
        

    # Functions : 與元件連結的函式功能
    def pushButton_train_click(self):
        self.trainer = Trainer()
        self.trainer.MainWindow.showMaximized()
        menu.MainWindow.hide()

    def pushButton_verify_click(self):
        self.verifier = Verifier()
        self.verifier.MainWindow.showMaximized()
        menu.MainWindow.hide()

    def pushButton_detect_click(self):
        self.detector = Detector()
        self.detector.MainWindow.showMaximized()
        menu.MainWindow.hide()


# 程式進入點
app = QApplication([])
menu = Menu()
menu.MainWindow.show()
app.exec_()