# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'menu.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.setEnabled(True)
        MainWindow.resize(344, 475)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: rgb(0, 85, 0);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_train = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_train.setGeometry(QtCore.QRect(80, 120, 181, 85))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_train.setFont(font)
        self.pushButton_train.setStyleSheet("QPushButton\n"
"{\n"
"    border-image:url(:./ui_img/menu_train.png)\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    border-image:url(:./ui_img/menu_train_2.png)\n"
"}")
        self.pushButton_train.setText("")
        self.pushButton_train.setObjectName("pushButton_train")
        self.pushButton_verify = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_verify.setGeometry(QtCore.QRect(80, 210, 181, 85))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_verify.setFont(font)
        self.pushButton_verify.setStyleSheet("QPushButton\n"
"{\n"
"    border-image:url(:./ui_img/menu_verify.png)\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    border-image:url(:./ui_img/menu_verify_2.png)\n"
"}")
        self.pushButton_verify.setText("")
        self.pushButton_verify.setObjectName("pushButton_verify")
        self.pushButton_detect = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_detect.setGeometry(QtCore.QRect(80, 300, 181, 85))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_detect.setFont(font)
        self.pushButton_detect.setStyleSheet("QPushButton\n"
"{\n"
"    border-image:url(:./ui_img/menu_detect.png)\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    border-image:url(:./ui_img/menu_detect_2.png)\n"
"}")
        self.pushButton_detect.setText("")
        self.pushButton_detect.setObjectName("pushButton_detect")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CNN Template"))
import pic
