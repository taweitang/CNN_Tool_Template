import os
import time
from ui_interface import *
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import QThread
from models.config import config
from datetime import datetime
from os import listdir
import numpy as np
from functools import partial
import configparser
from verify_ui import Ui_MainWindow
from models.CNN import CNN_ResNet

class Verifier:
    
    # 建構子
    def __init__(self) -> None:
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        self.ui.textEdit_info.setLineWrapMode(QTextEdit.NoWrap)
        self.log = []
        self.datasetInfo = {}
        self.results = {}
        self.filelist = []
        self.timeflag = time.time()
        self.ui.listWidget.itemClicked.connect(self.listWidget_Clicked)
        self.thread = QThread()
        self.model = None
        self.threshold = 0
        self.tested = False
        self.labels = []
        self.scores = []
        self.borderlist = []
        self.now_image = ""
        self.isPC = False
        self.updateInfo('Verifier initialized successfully.')
        self.load_config() 
        self.update_class_list()


    # Event
        self.ui.pushButton_model_path.clicked.connect(self.select_model_file)
        self.ui.pushButton_run.clicked.connect(self.start_testing_thread)
        self.ui.pushButton_test_path.clicked.connect(self.select_test_folder)

    # Functions
    def init_config(self):
        config = configparser.ConfigParser()
        config['setting'] = {}
        config['setting']['textEdit_model_path'] = ''
        config['setting']['textEdit_testdata'] = ''
        with open('verify.ini', 'w') as configfile:
            config.write(configfile)

    def load_config(self):
        if not os.path.exists('verify.ini'):
            self.init_config()
        config = configparser.ConfigParser()
        config.sections()
        config.read('verify.ini')
        self.ui.textEdit_model_path.setPlainText(config['setting']['textEdit_model_path'])
        self.ui.textEdit_testdata.setPlainText(config['setting']['textEdit_testdata'])

    def save_config(self):
        config = configparser.ConfigParser()
        config['setting'] = {}
        config['setting']['textEdit_model_path'] = self.ui.textEdit_model_path.toPlainText()
        config['setting']['textEdit_testdata'] = self.ui.textEdit_testdata.toPlainText()
        with open('verify.ini', 'w') as configfile:
            config.write(configfile)

    def refreshACC(self):
        correct = 0
        for k in self.results:
            #print(self.results[k])
            if self.results[k]['label'] == self.results[k]['result']:
                correct +=1
            accuracy = correct / len(self.results)
        self.ui.label_acc.setText(str(accuracy)[:4] + '/1.00')
        progressBarVal = int(accuracy*441)
        self.ui.progressBar_acc.resize(progressBarVal,16)

    def refresh_images_grid(self):
        # Clear Layout
        for i in reversed(range(self.ui.gridLayout_images.count())): 
            self.ui.gridLayout_images.itemAt(i).widget().setParent(None)
        self.buttonlist = []
        self.borderlist = []

        len_list = len(self.filelist)
        n_cols = 6
        n_rows = len_list//n_cols +1
        i = 0
        for row in range(n_rows): 
           for column in range(n_cols): 
                border = QWidget()
                border.setMinimumSize(QSize(100, 100))
                border.setMaximumSize(QSize(100, 100))
                button = QPushButton('', border)
                button.setMinimumSize(QSize(90, 90))
                button.setMaximumSize(QSize(90, 90))
                button.move(5, 5)
                button.setStyleSheet("border-image : url({0});".format(self.filelist[i]))
                button.clicked.connect(partial(self.image_on_clicked,i))
                self.ui.gridLayout_images.addWidget(border, row+1, column)
                self.borderlist.append(border)
                i += 1
                if i == len_list: break

    def refresh_image_main(self):
        self.pixmap = QPixmap(self.now_image)
        self.pixmap = self.pixmap.scaledToHeight(400)
        width = self.pixmap.width()
        margin = (700-width)//2
        if margin<0:
            margin = 0
        self.ui.widget_image_layout.setContentsMargins(margin,14,margin,14)
        self.ui.label_image.setPixmap(self.pixmap)
    
    def image_on_clicked(self, i):
        file_path = self.filelist[i]
        self.now_image = file_path
        self.refresh_image_main()
        if(self.tested):
            info = self.results[file_path]
            self.ui.textEdit_result.setPlainText('')
            self.ui.textEdit_result.append('File Name: ' + file_path.split('/')[-1])
            self.ui.textEdit_result.append('Label: ' + info['label'])
            self.ui.textEdit_result.append('Predict: ' + info['result'])
            self.ui.textEdit_result.append('Score: ' + str(info['score']))

    def select_test_folder(self):
        defult = self.ui.textEdit_testdata.toPlainText()
        folderpath = ''
        if defult and defult!='':
            folderpath = QtWidgets.QFileDialog.getExistingDirectory(directory=defult)
        else:
            folderpath = QtWidgets.QFileDialog.getExistingDirectory()
        if folderpath and folderpath!='':
            self.ui.textEdit_testdata.setPlainText('')
            self.ui.textEdit_testdata.append(folderpath)
            self.update_class_list()

    def select_model_file(self):
        defult = self.ui.textEdit_model_path.toPlainText()
        fileName = ''
        if defult and defult!='':
            fileName,_ = QFileDialog.getOpenFileName(None,("Select Model"), defult, ("Model Files (*.pt)"))
        else:
            fileName,_ = QFileDialog.getOpenFileName(None,("Select Model"), "", ("Model Files (*.pt)"))
        if fileName and fileName!='':
            self.ui.textEdit_model_path.setPlainText('')
            self.ui.textEdit_model_path.append(fileName)
        
    def listWidget_Clicked(self, item):
        folder = self.ui.textEdit_testdata.toPlainText() + "/" + item.text()
        self.update_file_list(folder)
        self.refresh_images_grid()

    def update_class_list(self):
        root = self.ui.textEdit_testdata.toPlainText()
        if not os.path.exists(root):
            return
        self.classlist = []
        self.ui.listWidget.clear()
        for c in listdir(root):
            if os.path.isdir(root + '/' + c):
                self.classlist.append(root + '/' + c)
                self.ui.listWidget.addItem(c)  
        self.updateInfo('Loaded categories form: '+ root)   

    def update_file_list(self,folder):
        self.filelist = []
        for f in listdir(folder):
            if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp'):
                self.filelist.append(folder + '/' + f)
        self.updateInfo('Loaded files form: '+ folder)   

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

    def getResult(self,status):
        # image_path,result,score 
        status = status.split('^U^')
        if status[0]!='Done' and status[0]!='init':
            info = {}
            info['label'] = status[0].split("/")[-2]
            info['result'] = status[1]
            info['score'] = float(status[2])
            self.results[status[0]] = info
            self.ui.label_progress.setText(str(len(self.results)) + '/' + str(self.datasetInfo['n_items']))
            progressBarVal = int(len(self.results)/self.datasetInfo['n_items']*441)
            self.ui.progressBar_progress.resize(progressBarVal,16)

        elif status[0]=='Done':
            self.tested = True
            self.scores = np.array(self.scores)
            self.refreshACC()
            
        elif status[0]=='init':
            self.datasetInfo['n_items'] = int(status[2])
            self.results = {}

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
            

    def start_testing_thread(self):
        self.save_config()       
        self.datasetInfo = {}
        self.tested = False
        self.cfg = config()
        self.cfg.featmap_size = (224,224)
        self.cfg.train_data_path = ""
        self.cfg.test_data_path = self.ui.textEdit_testdata.toPlainText()
        self.cfg.model_name = self.ui.textEdit_model_path.toPlainText()
        self.model = CNN_ResNet(self.cfg)

        self.model.moveToThread(self.thread)
        self.thread.started.connect(self.model.verify)
        self.model.finished.connect(self.thread.quit)
        self.model.sendinfo.connect(self.updateInfo)
        self.model.sendresult.connect(self.getResult)
        self.thread.start()


#app = QApplication([])
#verifier = Verifier()
#verifier.MainWindow.show()
#app.exec_()