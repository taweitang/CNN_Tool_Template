import os
import time
from ui_interface import *
from PyQt5 import QtWidgets
from PyQt5.QtChart import QChart,QLineSeries,QChartView
from PyQt5.QtGui import QPainter,QPixmap
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import QThread
from models.config import config
from datetime import datetime
import os
import shutil
import configparser
from train_ui import Ui_MainWindow
from models.CNN import CNN_ResNet

class Trainer:
    
    def __init__(self) -> None:
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        self.ui.textEdit_info.setLineWrapMode(QTextEdit.NoWrap)
        self.log = []
        self.initStatus = {}
        self.series_loss = QLineSeries()
        self.series_acc = QLineSeries()
        self.loss = []
        self.auc = []
        self.filelist = []
        self.timeflag = time.time()
        self.ui.listWidget.itemClicked.connect(self.listWidget_Clicked)
        self.thread = QThread()
        self.model = None
        self.updateInfo('Trainer initialized successfully.')
        self.load_config()   
        self.init_training_chart()
        self.update_file_list()
        self.comboBox_model_change()

    # Event (連結UI元件與事件)
        self.ui.pushButton_run.clicked.connect(self.start_training_thread)
        self.ui.pushButton_train_path.clicked.connect(self.select_train_folder)
        self.ui.pushButton_test_path.clicked.connect(self.select_test_folder)
        self.ui.pushButton_output_path.clicked.connect(self.select_save_folder)
        self.ui.comboBox_model.currentIndexChanged.connect(self.comboBox_model_change)

    # Functions
    def init_config(self):
        config = configparser.ConfigParser()
        config['setting'] = {}
        config['setting']['comboBox_model'] = '0'
        config['setting']['textEdit_epoch'] = '10'
        config['setting']['textEdit_length'] = '256'
        config['setting']['textEdit_lr'] = '1.0'
        config['setting']['comboBox_lr'] = '0'
        config['setting']['textEdit_traindata'] = ''
        config['setting']['textEdit_testdata'] = ''
        config['setting']['textEdit_output'] = ''
        with open('train.ini', 'w') as configfile:
            config.write(configfile)

    def load_config(self):
        if not os.path.exists('train.ini'):
            self.init_config()
        config = configparser.ConfigParser()
        config.sections()
        config.read('train.ini')
        self.ui.comboBox_model.setCurrentIndex(int(config['setting']['comboBox_model']))
        self.ui.textEdit_epoch.setPlainText(config['setting']['textEdit_epoch'])
        self.ui.textEdit_length.setPlainText(config['setting']['textEdit_length'])
        self.ui.textEdit_lr.setPlainText(config['setting']['textEdit_lr'])
        self.ui.comboBox_lr.setCurrentIndex(int(config['setting']['comboBox_lr']))
        self.ui.textEdit_traindata.setPlainText(config['setting']['textEdit_traindata'])
        self.ui.textEdit_testdata.setPlainText(config['setting']['textEdit_testdata'])
        self.ui.textEdit_output.setPlainText(config['setting']['textEdit_output'])

    def save_config(self):
        config = configparser.ConfigParser()
        config['setting'] = {}
        config['setting']['comboBox_model'] = str(self.ui.comboBox_model.currentIndex())
        config['setting']['textEdit_epoch'] = self.ui.textEdit_epoch.toPlainText()
        config['setting']['textEdit_length'] = self.ui.textEdit_length.toPlainText()
        config['setting']['textEdit_lr'] = self.ui.textEdit_lr.toPlainText()
        config['setting']['comboBox_lr'] = str(self.ui.comboBox_lr.currentIndex())
        config['setting']['textEdit_traindata'] = self.ui.textEdit_traindata.toPlainText()
        config['setting']['textEdit_testdata'] = self.ui.textEdit_testdata.toPlainText()
        config['setting']['textEdit_output'] = self.ui.textEdit_output.toPlainText()
        with open('train.ini', 'w') as configfile:
            config.write(configfile)

    def init_training_chart(self):
        self.chart = QChart()
        self.chart.legend().hide()
        self.chart.addSeries(self.series_loss)
        self.chart.createDefaultAxes()
        self.chartView = QChartView(self.chart)
        self.chartView.setRenderHint(QPainter.Antialiasing)
        self.chartView.chart().setTheme(QChart.ChartThemeDark)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.chartView.sizePolicy().hasHeightForWidth())
        self.chartView.setSizePolicy(sizePolicy)
        self.chartView.setMinimumSize(QSize(0, 300))
        self.ui.line_charts_cont.addWidget(self.chartView, 0, 0, 9, 9)

    def clear_charts(self):
        for i in reversed(range(self.ui.line_charts_cont.count())): 
            self.ui.line_charts_cont.itemAt(i).widget().setParent(None)

    def select_train_folder(self):
        defult = self.ui.textEdit_traindata.toPlainText()
        folderpath = ''
        if defult and defult!='':
            folderpath = QtWidgets.QFileDialog.getExistingDirectory(directory = defult)
        else:
            folderpath = QtWidgets.QFileDialog.getExistingDirectory()
        if folderpath and folderpath!='':
            self.ui.textEdit_traindata.setPlainText('')
            self.ui.textEdit_traindata.append(folderpath)
            self.update_file_list()

    def select_test_folder(self):
        defult = self.ui.textEdit_testdata.toPlainText()
        folderpath = ''
        if defult and defult!='':
            folderpath = QtWidgets.QFileDialog.getExistingDirectory(directory = defult)
        else:
            folderpath = QtWidgets.QFileDialog.getExistingDirectory()
        if folderpath and folderpath!='':
            self.ui.textEdit_testdata.setPlainText('')
            self.ui.textEdit_testdata.append(folderpath)

    def select_save_folder(self):
        defult = self.ui.textEdit_output.toPlainText()
        folderpath = ''
        if defult and defult!='':
            folderpath = QtWidgets.QFileDialog.getExistingDirectory(directory = defult)
        else:
            folderpath = QtWidgets.QFileDialog.getExistingDirectory()
        if folderpath and folderpath!='':
            self.ui.textEdit_output.setPlainText('')
            self.ui.textEdit_output.append(folderpath)
        
    def listWidget_Clicked(self, item):
        file_path = item.text()
        self.pixmap = QPixmap(file_path)
        self.pixmap = self.pixmap.scaledToHeight(400)
        width = self.pixmap.width()
        margin = (700-width)//2
        if margin<0:
            margin = 0
        self.ui.widget_image_layout.setContentsMargins(margin,14,margin,14)
        self.ui.label_image.setPixmap(self.pixmap)

    def update_file_list(self):
        folder = self.ui.textEdit_traindata.toPlainText()
        if not os.path.exists(folder):
            return
        self.filelist = []
        self.ui.listWidget.clear()
        #for f in listdir(folder):
        for root, _,files  in os.walk(folder):
            for f in files:
                if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp'):
                    self.filelist.append(os.path.join(root,f))
                    self.ui.listWidget.addItem(os.path.join(root,f))  
        self.updateInfo('Loaded files form: '+ folder)   
    
    def update_chart(self,xmax,isloss=True):
        if isloss:
            ax1 = self.chart.axisX(self.series_loss)
            ay1 = self.chart.axisY(self.series_loss)
            ax1.setMin(0)
            ax1.setMax(xmax)
            ay1.setMin(0)
            ay1.setMax(max(self.loss))
        else:
            ax1 = self.chart.axisX(self.series_acc)
            ay1 = self.chart.axisY(self.series_acc)
            ax1.setMin(0)
            ax1.setMax(xmax)
            ay1.setMin(0)
            ay1.setMax(max(self.loss))
        
    def updateInfo(self,info):           
        now = datetime.now().strftime("%H:%M:%S")
        info = '[{}] '.format(now) + info
        if len(info)>80:
            info = info[:80] + '...'
        self.log.append(info)
        if len(self.log)==4:
            self.log = self.log[1:4]
        if len(self.log)>0:
            self.ui.label_info_1.setText(self.log[0])
        if len(self.log)>1:
            self.ui.label_info_2.setText(self.log[1])
        if len(self.log)>2:
            self.ui.label_info_3.setText(self.log[2])

    # 接收背景訓練模型的執行續回傳的資訊
    def updateProcess(self,status):
        status = status.split(',')
        if status[-1] =='*':
            # init
            self.initStatus['epoch'] = int(status[0])
            self.initStatus['step'] =  int(status[1])
            self.initStatus['time'] = 0
            self.timeflag = time.time()
        elif status[-1] =='0':
            # update step
            # format: [epoch,step,loss,auc]
            # chart
            self.series_loss.append(int(status[1]), float(status[2]))
            self.loss.append(float(status[2]))
            self.update_chart(int(status[1]))

            #self.chart.addSeries(self.series_loss)
            # epoch bar
            self.ui.label_epoch.setText(status[0] + '/' + str(self.initStatus['epoch']))
            epochBarVal = int(float(status[0])/self.initStatus['epoch']*441)
            #print('epoch bar:',epochBarVal)
            self.ui.progressBar_epoch.resize(epochBarVal,16)
            self.loss.append(float(status[2]))
            # time calculate
            now = time.time()
            time_this_step = now - self.timeflag
            self.timeflag = now
            if self.initStatus['time'] == 0:
                self.initStatus['time'] = time_this_step    
            else:
                self.initStatus['time'] = 0.9*self.initStatus['time'] + 0.1*time_this_step
            remain_time = self.initStatus['time']*(self.initStatus['step']-int(status[1]))
            remain_time_str = ''
            remain_time_str+= str(int(remain_time//3600))+':'
            remain_time %=3600
            temp = str(int(remain_time//60))
            if len(temp)==1:
                temp = '0'+temp
            remain_time_str+= temp+':'
            remain_time %=60
            temp = str(int(remain_time))
            if len(temp)==1:
                temp = '0'+temp
            remain_time_str+= temp
            self.ui.label_time.setText(remain_time_str)
            timeBarVal = int(((self.initStatus['step']-int(status[1]))/self.initStatus['step'])*441)
            self.ui.progressBar_time.resize(timeBarVal,16)
        else:
            # upadte auc
            self.ui.label_AUC.setText(status[3][:4] + '/1.00')
            AUCBarVal = int(float(status[3])*441)
            self.ui.progressBar_AUC.resize(AUCBarVal,16)
            self.auc.append(float(status[3]))

    def enable_ui(self,enable):
        #Button
        self.ui.pushButton_run.setEnabled(enable)
        self.ui.pushButton_train_path.setEnabled(enable)
        self.ui.pushButton_test_path.setEnabled(enable)
        self.ui.pushButton_output_path.setEnabled(enable)
        #Combobox
        self.ui.comboBox_model.setEnabled(enable)
        self.ui.comboBox_lr.setEnabled(enable)
        #TextEdit
        self.ui.textEdit_epoch.setEnabled(enable)
        self.ui.textEdit_length.setEnabled(enable)
        self.ui.textEdit_lr.setEnabled(enable)

    def comboBox_model_change(self):
        self.ui.label.setText('Training Epoch')
        self.ui.label_3.setText('Learning Rate')
        self.ui.label_11.setText('Remain time')
        self.ui.label_12.setText('Training epoch')
        self.ui.label_time.setText('0:0:0')
        self.ui.label_epoch.setText('0/0')
        
    def start_training_thread(self):
        self.save_config()
        self.enable_ui(False)
        self.initStatus = {}
        self.loss = []
        self.auc = []
        model_index = self.ui.comboBox_model.currentIndex()
        self.cfg = config()
        self.cfg.model_index = model_index
        self.cfg.epochs = int(self.ui.textEdit_epoch.toPlainText())
        self.ui.label_epoch.setText('0/' + self.ui.textEdit_epoch.toPlainText())
        self.ui.label_time.setText('0:0:0')
        self.ui.label_AUC.setText('0.00/1.00')
        self.ui.progressBar_epoch.resize(0,16)
        self.ui.progressBar_time.resize(0,16)
        self.ui.progressBar_AUC.resize(0,16)
        self.cfg.featmap_size = (int(self.ui.textEdit_length.toPlainText()),int(self.ui.textEdit_length.toPlainText()))
        self.cfg.lr = float(self.ui.textEdit_lr.toPlainText()) * 10**((-1)*(self.ui.comboBox_lr.currentIndex()+3))
        self.cfg.train_data_path = self.ui.textEdit_traindata.toPlainText()
        self.cfg.test_data_path = self.ui.textEdit_testdata.toPlainText()
        self.cfg.save_path = self.ui.textEdit_output.toPlainText()
        self.cfg.data_name = datetime.now().strftime("%Y%m%d%H%M%S")
        self.model = CNN_ResNet(self.cfg)
        self.model.moveToThread(self.thread)
        self.thread.started.connect(self.model.train)
        self.model.finished.connect(self.thread.quit)
        self.model.sendinfo.connect(self.updateInfo)
        self.model.progress.connect(self.updateProcess)
        self.thread.start()


#app = QApplication([])
#trainer = Trainer()
#trainer.MainWindow.show()
#app.exec_()