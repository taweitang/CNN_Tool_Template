# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'train.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1916, 1016)
        MainWindow.setStyleSheet("border-image: url(:/ui_img/background.jpg);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setContentsMargins(20, 20, 20, 20)
        self.verticalLayout_4.setSpacing(20)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(20)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setStyleSheet("border-image: url(:/ui_img/trans.png);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 382, 202))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.widget_8 = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        self.widget_8.setStyleSheet("border-image: url(:/ui_img/train_ADModel.png);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.widget_8.setObjectName("widget_8")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.widget_8)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.widget = QtWidgets.QWidget(self.widget_8)
        self.widget.setMinimumSize(QtCore.QSize(382, 202))
        self.widget.setStyleSheet("border-image: url(:/ui_img/trans.png);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.widget.setObjectName("widget")
        self.comboBox_model = QtWidgets.QComboBox(self.widget)
        self.comboBox_model.setGeometry(QtCore.QRect(140, 90, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(16)
        self.comboBox_model.setFont(font)
        self.comboBox_model.setAutoFillBackground(False)
        self.comboBox_model.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"border: 1px solid rgb(68,114,196);\n"
"color: rgb(255,255,255);")
        self.comboBox_model.setFrame(True)
        self.comboBox_model.setObjectName("comboBox_model")
        self.comboBox_model.addItem("")
        self.label_14 = QtWidgets.QLabel(self.widget)
        self.label_14.setGeometry(QtCore.QRect(30, 80, 71, 51))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(50)
        self.label_14.setFont(font)
        self.label_14.setStyleSheet("color: rgb(255,255,255);")
        self.label_14.setLineWidth(6)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_6.addWidget(self.widget)
        self.horizontalLayout_5.addWidget(self.widget_8)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout.addWidget(self.scrollArea)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.scrollArea_2 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_2.setStyleSheet("border-image: url(:/ui_img/trans.png);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.scrollArea_2.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 1468, 202))
        self.scrollAreaWidgetContents_2.setAutoFillBackground(False)
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.widget_7 = QtWidgets.QWidget(self.scrollAreaWidgetContents_2)
        self.widget_7.setMinimumSize(QtCore.QSize(1468, 202))
        self.widget_7.setStyleSheet("border-image: url(:/ui_img/train_detail.png);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.widget_7.setObjectName("widget_7")
        self.widget_2 = QtWidgets.QWidget(self.widget_7)
        self.widget_2.setGeometry(QtCore.QRect(0, 0, 1472, 207))
        self.widget_2.setMinimumSize(QtCore.QSize(1468, 202))
        self.widget_2.setAutoFillBackground(False)
        self.widget_2.setStyleSheet("border-image: url(:/ui_img/train.png);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.widget_2.setObjectName("widget_2")
        self.label = QtWidgets.QLabel(self.widget_2)
        self.label.setGeometry(QtCore.QRect(50, 55, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(255,255,255);")
        self.label.setObjectName("label")
        self.textEdit_epoch = QtWidgets.QTextEdit(self.widget_2)
        self.textEdit_epoch.setGeometry(QtCore.QRect(200, 60, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(13)
        self.textEdit_epoch.setFont(font)
        self.textEdit_epoch.setStyleSheet("border: 1px solid rgb(68,196,114);\n"
"color: rgb(255,255,255);")
        self.textEdit_epoch.setObjectName("textEdit_epoch")
        self.label_2 = QtWidgets.QLabel(self.widget_2)
        self.label_2.setGeometry(QtCore.QRect(50, 100, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: rgb(255,255,255);")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.widget_2)
        self.label_3.setGeometry(QtCore.QRect(50, 145, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: rgb(255,255,255);")
        self.label_3.setObjectName("label_3")
        self.textEdit_lr = QtWidgets.QTextEdit(self.widget_2)
        self.textEdit_lr.setGeometry(QtCore.QRect(200, 150, 101, 41))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(13)
        self.textEdit_lr.setFont(font)
        self.textEdit_lr.setStyleSheet("border: 1px solid rgb(68,196,114);\n"
"color: rgb(255,255,255);")
        self.textEdit_lr.setObjectName("textEdit_lr")
        self.textEdit_length = QtWidgets.QTextEdit(self.widget_2)
        self.textEdit_length.setEnabled(False)
        self.textEdit_length.setGeometry(QtCore.QRect(200, 105, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(13)
        self.textEdit_length.setFont(font)
        self.textEdit_length.setStyleSheet("border: 1px solid rgb(68,196,114);\n"
"color: rgb(255,255,255);")
        self.textEdit_length.setObjectName("textEdit_length")
        self.label_7 = QtWidgets.QLabel(self.widget_2)
        self.label_7.setGeometry(QtCore.QRect(500, 55, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setAutoFillBackground(False)
        self.label_7.setStyleSheet("color: rgb(255,255,255);")
        self.label_7.setObjectName("label_7")
        self.textEdit_traindata = QtWidgets.QTextEdit(self.widget_2)
        self.textEdit_traindata.setEnabled(False)
        self.textEdit_traindata.setGeometry(QtCore.QRect(710, 60, 661, 41))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(13)
        self.textEdit_traindata.setFont(font)
        self.textEdit_traindata.setStyleSheet("border: 1px solid rgb(68,196,114);\n"
"color: rgb(255,255,255);")
        self.textEdit_traindata.setObjectName("textEdit_traindata")
        self.textEdit_testdata = QtWidgets.QTextEdit(self.widget_2)
        self.textEdit_testdata.setEnabled(False)
        self.textEdit_testdata.setGeometry(QtCore.QRect(710, 105, 661, 41))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(13)
        self.textEdit_testdata.setFont(font)
        self.textEdit_testdata.setStyleSheet("border: 1px solid rgb(68,196,114);\n"
"color: rgb(255,255,255);")
        self.textEdit_testdata.setObjectName("textEdit_testdata")
        self.label_8 = QtWidgets.QLabel(self.widget_2)
        self.label_8.setGeometry(QtCore.QRect(500, 145, 181, 51))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("color: rgb(255,255,255);")
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.widget_2)
        self.label_9.setGeometry(QtCore.QRect(500, 100, 191, 51))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("color: rgb(255,255,255);")
        self.label_9.setObjectName("label_9")
        self.textEdit_output = QtWidgets.QTextEdit(self.widget_2)
        self.textEdit_output.setEnabled(False)
        self.textEdit_output.setGeometry(QtCore.QRect(710, 150, 661, 41))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(13)
        self.textEdit_output.setFont(font)
        self.textEdit_output.setStyleSheet("border: 1px solid rgb(68,196,114);\n"
"color: rgb(255,255,255);")
        self.textEdit_output.setObjectName("textEdit_output")
        self.pushButton_train_path = QtWidgets.QPushButton(self.widget_2)
        self.pushButton_train_path.setGeometry(QtCore.QRect(1400, 62, 31, 31))
        self.pushButton_train_path.setStyleSheet("QPushButton\n"
"{\n"
"    border-image:url(:./ui_img/open.png);\n"
"    background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    border-image:url(:./ui_img/open2.png);\n"
"    background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.pushButton_train_path.setText("")
        self.pushButton_train_path.setObjectName("pushButton_train_path")
        self.pushButton_test_path = QtWidgets.QPushButton(self.widget_2)
        self.pushButton_test_path.setGeometry(QtCore.QRect(1400, 108, 31, 31))
        self.pushButton_test_path.setStyleSheet("QPushButton\n"
"{\n"
"    border-image:url(:./ui_img/open.png);\n"
"    background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    border-image:url(:./ui_img/open2.png);\n"
"    background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.pushButton_test_path.setText("")
        self.pushButton_test_path.setObjectName("pushButton_test_path")
        self.pushButton_output_path = QtWidgets.QPushButton(self.widget_2)
        self.pushButton_output_path.setGeometry(QtCore.QRect(1400, 155, 31, 31))
        self.pushButton_output_path.setStyleSheet("QPushButton\n"
"{\n"
"    border-image:url(:./ui_img/open.png);\n"
"    background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    border-image:url(:./ui_img/open2.png);\n"
"    background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.pushButton_output_path.setText("")
        self.pushButton_output_path.setObjectName("pushButton_output_path")
        self.comboBox_lr = QtWidgets.QComboBox(self.widget_2)
        self.comboBox_lr.setGeometry(QtCore.QRect(350, 150, 101, 41))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(13)
        self.comboBox_lr.setFont(font)
        self.comboBox_lr.setAutoFillBackground(False)
        self.comboBox_lr.setStyleSheet("border: 1px solid rgb(68,196,114);\n"
"color: rgb(255,255,255);")
        self.comboBox_lr.setFrame(True)
        self.comboBox_lr.setObjectName("comboBox_lr")
        self.comboBox_lr.addItem("")
        self.comboBox_lr.addItem("")
        self.comboBox_lr.addItem("")
        self.comboBox_lr.addItem("")
        self.label_10 = QtWidgets.QLabel(self.widget_2)
        self.label_10.setGeometry(QtCore.QRect(320, 150, 16, 41))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.label_10.setFont(font)
        self.label_10.setStyleSheet("color: rgb(255,255,255);")
        self.label_10.setObjectName("label_10")
        self.verticalLayout_5.addWidget(self.widget_7)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.verticalLayout.addWidget(self.scrollArea_2)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout_2.setStretch(0, 66)
        self.horizontalLayout_2.setStretch(1, 253)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(20)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(20)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.scrollArea_3 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_3.setStyleSheet("border-image: url(:/ui_img/trans.png);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollArea_3.setObjectName("scrollArea_3")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 103, 744))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents_3)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.widget_9 = QtWidgets.QWidget(self.scrollAreaWidgetContents_3)
        self.widget_9.setMinimumSize(QtCore.QSize(103, 744))
        self.widget_9.setStyleSheet("border-image: url(:/ui_img/train_tool.png);")
        self.widget_9.setObjectName("widget_9")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.widget_9)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.widget_3 = QtWidgets.QWidget(self.widget_9)
        self.widget_3.setStyleSheet("border-image: url(:/ui_img/train.png);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.widget_3.setObjectName("widget_3")
        self.pushButton_run = QtWidgets.QPushButton(self.widget_3)
        self.pushButton_run.setGeometry(QtCore.QRect(20, 70, 61, 61))
        self.pushButton_run.setStyleSheet("QPushButton\n"
"{\n"
"    border-image:url(:./ui_img/run.png);\n"
"    background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    border-image:url(:./ui_img/run2.png);\n"
"    background-color: rgba(255, 255, 255, 0);\n"
"}")
        self.pushButton_run.setText("")
        self.pushButton_run.setObjectName("pushButton_run")
        self.horizontalLayout_8.addWidget(self.widget_3)
        self.horizontalLayout_7.addWidget(self.widget_9)
        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)
        self.verticalLayout_2.addWidget(self.scrollArea_3)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.scrollArea_6 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_6.setStyleSheet("border-image: url(:/ui_img/trans.png);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.scrollArea_6.setWidgetResizable(True)
        self.scrollArea_6.setObjectName("scrollArea_6")
        self.scrollAreaWidgetContents_6 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_6.setGeometry(QtCore.QRect(0, 0, 255, 744))
        self.scrollAreaWidgetContents_6.setObjectName("scrollAreaWidgetContents_6")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents_6)
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_12.setSpacing(0)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.widget_4 = QtWidgets.QWidget(self.scrollAreaWidgetContents_6)
        self.widget_4.setMinimumSize(QtCore.QSize(255, 744))
        self.widget_4.setStyleSheet("border-image: url(:/ui_img/train_list.png);")
        self.widget_4.setObjectName("widget_4")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.widget_4)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.widget_11 = QtWidgets.QWidget(self.widget_4)
        self.widget_11.setStyleSheet("border-image: url(:/ui_img/train.png);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.widget_11.setObjectName("widget_11")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.widget_11)
        self.verticalLayout_8.setContentsMargins(15, 60, 15, 20)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.listWidget = QtWidgets.QListWidget(self.widget_11)
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(13)
        self.listWidget.setFont(font)
        self.listWidget.setStyleSheet("border: 1px solid rgba(68,114,196,0);\n"
"color: rgb(255,255,255);")
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout_8.addWidget(self.listWidget)
        self.verticalLayout_7.addWidget(self.widget_11)
        self.horizontalLayout_12.addWidget(self.widget_4)
        self.scrollArea_6.setWidget(self.scrollAreaWidgetContents_6)
        self.verticalLayout_3.addWidget(self.scrollArea_6)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.horizontalLayout_3.setStretch(0, 18)
        self.horizontalLayout_3.setStretch(1, 44)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_3)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setSpacing(40)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.scrollArea_5 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_5.setStyleSheet("border-image: url(:/ui_img/trans.png);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.scrollArea_5.setWidgetResizable(True)
        self.scrollArea_5.setObjectName("scrollArea_5")
        self.scrollAreaWidgetContents_5 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_5.setGeometry(QtCore.QRect(0, 0, 1468, 535))
        self.scrollAreaWidgetContents_5.setObjectName("scrollAreaWidgetContents_5")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents_5)
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_11.setSpacing(0)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.widget_5 = QtWidgets.QWidget(self.scrollAreaWidgetContents_5)
        self.widget_5.setMinimumSize(QtCore.QSize(1468, 535))
        self.widget_5.setStyleSheet("border-image: url(:/ui_img/train_vis.png);")
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.widget_5)
        self.horizontalLayout_13.setContentsMargins(0, 48, 0, 25)
        self.horizontalLayout_13.setSpacing(0)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.widget_image = QtWidgets.QWidget(self.widget_5)
        self.widget_image.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.widget_image.setStyleSheet("border-image: url(:/ui_img/train.png);")
        self.widget_image.setObjectName("widget_image")
        self.widget_image_layout = QtWidgets.QGridLayout(self.widget_image)
        self.widget_image_layout.setContentsMargins(150, 14, 150, 14)
        self.widget_image_layout.setObjectName("widget_image_layout")
        self.label_image = QtWidgets.QLabel(self.widget_image)
        self.label_image.setText("")
        self.label_image.setObjectName("label_image")
        self.widget_image_layout.addWidget(self.label_image, 0, 0, 1, 1)
        self.horizontalLayout_13.addWidget(self.widget_image)
        self.line_charts = QtWidgets.QWidget(self.widget_5)
        self.line_charts.setStyleSheet("border-image: url(:/ui_img/train.png);")
        self.line_charts.setObjectName("line_charts")
        self.line_charts_cont = QtWidgets.QGridLayout(self.line_charts)
        self.line_charts_cont.setObjectName("line_charts_cont")
        self.horizontalLayout_13.addWidget(self.line_charts)
        self.horizontalLayout_11.addWidget(self.widget_5)
        self.scrollArea_5.setWidget(self.scrollAreaWidgetContents_5)
        self.verticalLayout_6.addWidget(self.scrollArea_5)
        self.scrollArea_4 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_4.setStyleSheet("border-image: url(:/ui_img/trans.png);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollArea_4.setObjectName("scrollArea_4")
        self.scrollAreaWidgetContents_4 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_4.setGeometry(QtCore.QRect(0, 0, 1468, 171))
        self.scrollAreaWidgetContents_4.setObjectName("scrollAreaWidgetContents_4")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents_4)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.widget_10 = QtWidgets.QWidget(self.scrollAreaWidgetContents_4)
        self.widget_10.setMinimumSize(QtCore.QSize(1468, 171))
        self.widget_10.setStyleSheet("border-image: url(:/ui_img/train_info.png);")
        self.widget_10.setObjectName("widget_10")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.widget_10)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setSpacing(0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.widget_6 = QtWidgets.QWidget(self.widget_10)
        self.widget_6.setStyleSheet("border-image: url(:/ui_img/train.png);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.widget_6.setObjectName("widget_6")
        self.textEdit_info = QtWidgets.QTextEdit(self.widget_6)
        self.textEdit_info.setEnabled(False)
        self.textEdit_info.setGeometry(QtCore.QRect(40, 65, 681, 91))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.textEdit_info.setFont(font)
        self.textEdit_info.setStyleSheet("border: 1px solid rgb(68,196,114);\n"
"color: rgb(255,255,255);")
        self.textEdit_info.setObjectName("textEdit_info")
        self.label_11 = QtWidgets.QLabel(self.widget_6)
        self.label_11.setGeometry(QtCore.QRect(760, 55, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_11.setFont(font)
        self.label_11.setStyleSheet("color: rgb(255,255,255);")
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.widget_6)
        self.label_12.setGeometry(QtCore.QRect(760, 84, 191, 51))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_12.setFont(font)
        self.label_12.setStyleSheet("color: rgb(255,255,255);")
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.widget_6)
        self.label_13.setGeometry(QtCore.QRect(760, 113, 181, 51))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_13.setFont(font)
        self.label_13.setStyleSheet("color: rgb(255,255,255);")
        self.label_13.setObjectName("label_13")
        self.label_epoch = QtWidgets.QLabel(self.widget_6)
        self.label_epoch.setGeometry(QtCore.QRect(1360, 85, 191, 51))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_epoch.setFont(font)
        self.label_epoch.setStyleSheet("color: rgb(255,255,255);")
        self.label_epoch.setObjectName("label_epoch")
        self.label_time = QtWidgets.QLabel(self.widget_6)
        self.label_time.setGeometry(QtCore.QRect(1360, 57, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_time.setFont(font)
        self.label_time.setStyleSheet("color: rgb(255,255,255);")
        self.label_time.setObjectName("label_time")
        self.label_AUC = QtWidgets.QLabel(self.widget_6)
        self.label_AUC.setGeometry(QtCore.QRect(1360, 112, 181, 51))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_AUC.setFont(font)
        self.label_AUC.setStyleSheet("border-image: url(:/ui_img/trans.jpg);\n"
"color: rgb(255,255,255);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_AUC.setObjectName("label_AUC")
        self.label_4 = QtWidgets.QLabel(self.widget_6)
        self.label_4.setGeometry(QtCore.QRect(900, 75, 441, 16))
        self.label_4.setStyleSheet("background-color: rgb(30, 30, 30);")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.progressBar_time = QtWidgets.QLabel(self.widget_6)
        self.progressBar_time.setGeometry(QtCore.QRect(900, 75, 0, 16))
        self.progressBar_time.setStyleSheet(" background-color: rgb(255,255,255)")
        self.progressBar_time.setText("")
        self.progressBar_time.setObjectName("progressBar_time")
        self.label_5 = QtWidgets.QLabel(self.widget_6)
        self.label_5.setGeometry(QtCore.QRect(900, 102, 441, 16))
        self.label_5.setStyleSheet("background-color: rgb(30, 30, 30);")
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.widget_6)
        self.label_6.setGeometry(QtCore.QRect(900, 130, 441, 16))
        self.label_6.setStyleSheet("background-color: rgb(30, 30, 30);")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.progressBar_epoch = QtWidgets.QLabel(self.widget_6)
        self.progressBar_epoch.setGeometry(QtCore.QRect(900, 102, 0, 16))
        self.progressBar_epoch.setStyleSheet(" background-color: rgb(255,255,255)")
        self.progressBar_epoch.setText("")
        self.progressBar_epoch.setObjectName("progressBar_epoch")
        self.progressBar_AUC = QtWidgets.QLabel(self.widget_6)
        self.progressBar_AUC.setGeometry(QtCore.QRect(900, 130, 0, 16))
        self.progressBar_AUC.setStyleSheet(" background-color: rgb(255,255,255)")
        self.progressBar_AUC.setText("")
        self.progressBar_AUC.setObjectName("progressBar_AUC")
        self.label_info_3 = QtWidgets.QLabel(self.widget_6)
        self.label_info_3.setGeometry(QtCore.QRect(40, 120, 681, 31))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.label_info_3.setFont(font)
        self.label_info_3.setStyleSheet("border: 1px solid rgba(68,196,114,0);\n"
"color: rgb(255,255,255);\n"
"")
        self.label_info_3.setObjectName("label_info_3")
        self.label_info_1 = QtWidgets.QLabel(self.widget_6)
        self.label_info_1.setGeometry(QtCore.QRect(40, 70, 681, 31))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.label_info_1.setFont(font)
        self.label_info_1.setStyleSheet("border: 1px solid rgba(68,196,114,0);\n"
"color: rgb(255,255,255);\n"
"")
        self.label_info_1.setObjectName("label_info_1")
        self.label_info_2 = QtWidgets.QLabel(self.widget_6)
        self.label_info_2.setGeometry(QtCore.QRect(40, 95, 681, 31))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.label_info_2.setFont(font)
        self.label_info_2.setStyleSheet("border: 1px solid rgba(68,114,196,0);\n"
"color: rgb(255,255,255);\n"
"")
        self.label_info_2.setObjectName("label_info_2")
        self.horizontalLayout_10.addWidget(self.widget_6)
        self.horizontalLayout_9.addWidget(self.widget_10)
        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_4)
        self.verticalLayout_6.addWidget(self.scrollArea_4)
        self.verticalLayout_6.setStretch(0, 100)
        self.verticalLayout_6.setStretch(1, 32)
        self.horizontalLayout_4.addLayout(self.verticalLayout_6)
        self.horizontalLayout_4.setStretch(0, 66)
        self.horizontalLayout_4.setStretch(1, 253)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        self.verticalLayout_4.setStretch(0, 374)
        self.verticalLayout_4.setStretch(1, 1360)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.comboBox_lr.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CNN Template"))
        self.comboBox_model.setItemText(0, _translate("MainWindow", "ResNet50"))
        self.label_14.setText(_translate("MainWindow", "Model"))
        self.label.setText(_translate("MainWindow", "Training Epoch"))
        self.textEdit_epoch.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Dubai\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">10</p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Image Length"))
        self.label_3.setText(_translate("MainWindow", "Learning Rate"))
        self.textEdit_lr.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Dubai\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1.0</p></body></html>"))
        self.textEdit_length.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Dubai\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">224</p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "Training Data Folder"))
        self.textEdit_traindata.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Dubai\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">C:/Users/David/Desktop/crack/train/good</p></body></html>"))
        self.textEdit_testdata.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Dubai\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">C:/Users/David/Desktop/crack/test</p></body></html>"))
        self.label_8.setText(_translate("MainWindow", "Output Folder"))
        self.label_9.setText(_translate("MainWindow", "Testing Data Folder"))
        self.textEdit_output.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Dubai\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">C:/Users/David/Desktop/AnomalyTool/save/crack2</p></body></html>"))
        self.comboBox_lr.setItemText(0, _translate("MainWindow", "10^-3"))
        self.comboBox_lr.setItemText(1, _translate("MainWindow", "10^-4"))
        self.comboBox_lr.setItemText(2, _translate("MainWindow", "10^-5"))
        self.comboBox_lr.setItemText(3, _translate("MainWindow", "10^-6"))
        self.label_10.setText(_translate("MainWindow", "X"))
        self.textEdit_info.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Dubai\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label_11.setText(_translate("MainWindow", "Remain time"))
        self.label_12.setText(_translate("MainWindow", "Training epoch"))
        self.label_13.setText(_translate("MainWindow", "Testing ACC"))
        self.label_epoch.setText(_translate("MainWindow", "0/0"))
        self.label_time.setText(_translate("MainWindow", "0:0:0"))
        self.label_AUC.setText(_translate("MainWindow", "0/1.00"))
        self.label_info_3.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.label_info_1.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.label_info_2.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
import pic
