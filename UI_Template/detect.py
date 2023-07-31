import os
from ui_interface import *
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import QThread
from models.config import config
from datetime import datetime
from camera import Camera
import cv2
import PyQt5
import configparser
from detect_ui import Ui_MainWindow
from camsetting_ui import Ui_MainWindow_Cam
from models.CNN import CNN_ResNet

# 建構子
class Detector:
    def __init__(self) -> None:
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        self.ui.textEdit_info.setLineWrapMode(QTextEdit.NoWrap)
        self.log = []
        self.datasetInfo = {}
        self.ProcessCam = Camera(index=0)
        if self.ProcessCam.connect:
            self.ProcessCam.rawdata.connect(self.show_webcam)
        self.init_cam()
        self.snapImage = None
        self.threshold = 0
        self.pause = True
        self.model_change = True
        self.ai_pause = True
        self.client_imgs = []
        self.thread = QThread()
        self.updateInfo('Detector initialized successfully.')
        self.load_config() 

    # Event
        self.ui.pushButton_model_path.clicked.connect(self.select_model_file)
        self.ui.pushButton_run_AI.clicked.connect(self.start_testing_thread)
        self.ui.pushButton_run_cam.clicked.connect(self.cam_run2)
        self.ui.comboBox_mode.currentIndexChanged.connect(self.comboBox_mode_change)

    # Functions
    def init_config(self):
        config = configparser.ConfigParser()
        config['setting'] = {}
        config['setting']['textEdit_model_path'] = ''
        with open('detect.ini', 'w') as configfile:
            config.write(configfile)

    def load_config(self):
        if not os.path.exists('detect.ini'):
            self.init_config()
        config = configparser.ConfigParser()
        config.sections()
        config.read('detect.ini')
        self.ui.textEdit_model_path.setPlainText(config['setting']['textEdit_model_path'])

    def save_config(self):
        config = configparser.ConfigParser()
        config['setting'] = {}
        config['setting']['textEdit_model_path'] = self.ui.textEdit_model_path.toPlainText()
        with open('detect.ini', 'w') as configfile:
            config.write(configfile)

    def init_cam(self):
        if self.ProcessCam.connect:  
            self.ProcessCam.open()   
            self.ProcessCam.start() 
            self.ProcessCam.set_update(True)

    def show_webcam(self, data):
        self.snapImage = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        self.label_show(self.ui.label_image, self.snapImage)

        if self.ai_pause:
            return
        if len(self.model.detect_imgs)==0:
            self.model.detect_imgs.append(self.snapImage)

    def padding_black(self, img,width,height):
        old_size = img.shape[:2]
        ratio = min(width/old_size[1],height/old_size[0])
        new_size = tuple([int(x*ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = width - new_size[1]
        delta_h = height - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=(0,0,0))

    def label_show(self,label,img):      
        if img is None or self.pause:
            return

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        _, _, channel = img.shape
        width = label.width()
        height = label.height()
        
        src_pv_img = self.padding_black(img, width, height)
        width = src_pv_img.shape[1]
        height = src_pv_img.shape[0]
        step = channel * width
        
        qImg = QImage(src_pv_img.data, width, height, step, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qImg)
        label.setPixmap(pix)
    
    def open_cam_setting(self):
        self.setting = CamSetting(self.ProcessCam.return_init())
        self.setting.exposure.connect(self.set_exposure)
        self.setting.focus.connect(self.set_focus)
        self.setting.w.connect(self.set_crop_w)
        self.setting.h.connect(self.set_crop_h)
        self.setting.rotate.connect(self.set_rotate)
        self.setting.flip.connect(self.set_flip)
        self.setting.cam_run.connect(self.cam_run)
        self.setting.run()
        self.updateInfo('Open cam setting.')   

    def cam_run(self,value):
        if value:
            self.pause = False
            self.ui.pushButton_run_cam.setStyleSheet("border-image : url(./ui_img/stop.png);")
            self.updateInfo('Cam start.')
        else:
            self.pause = True
            self.ui.pushButton_run_cam.setStyleSheet("border-image : url(./ui_img/run.png);")
            self.updateInfo('Cam stop.')
    
    def cam_run2(self):
        self.save_config()
        if self.pause:
            self.pause = False
            self.updateInfo('Cam start.')
            self.ui.pushButton_run_cam.setStyleSheet("border-image : url(./ui_img/stop.png);")
        else:
            self.pause = True
            self.updateInfo('Cam stop.')
            self.ui.pushButton_run_cam.setStyleSheet("border-image : url(./ui_img/run.png);")
        
    def set_exposure(self,value):
        self.ProcessCam.set_exposure(value)

    def set_focus(self,value):
        self.ProcessCam.set_focus(value)
    
    def set_crop_w(self,value):
        self.ProcessCam.set_crop_w(value)

    def set_crop_h(self,value):
        self.ProcessCam.set_crop_h(value)

    def set_rotate(self,value):
        self.ProcessCam.set_rotate(value)

    def set_flip(self,flips):
        self.ProcessCam.set_flip(flips[0],flips[1])

    def slider_value_changed(self,value):
        self.threshold = (value*self.detect_max + (100-value)*self.detect_min)/100
        self.ui.label_threshold.setText(str(round(self.threshold,8)))

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
            self.model_change = True

    def updateInfo(self,info):           
        now = datetime.now().strftime("%H:%M:%S")
        info = '[{}] '.format(now) + info
        if len(info)>94:
            info = info[:94] + '...'
        self.log.append(info)
        if len(self.log)==4:
            self.log = self.log[1:4]
        if len(self.log)>0:
            self.ui.label_info_1.setText(self.log[0])
        if len(self.log)>1:
            self.ui.label_info_2.setText(self.log[1])
        if len(self.log)>2:
            self.ui.label_info_3.setText(self.log[2])

    def comboBox_mode_change(self,value):
        self.pause = True
        self.ai_pause = True
        self.updateInfo('AI stop.')
        self.updateInfo('Cam mode.')
        self.ui.pushButton_run_AI.setStyleSheet("border-image : url(./ui_img/run_AI.png);")
        self.ui.pushButton_run_cam.setStyleSheet("border-image : url(./ui_img/run.png);")
        self.ui.label_7.setText('Device (Cam) Name')
        self.ui.textEdit_input.setText('Device0')
        self.ui.textEdit_input.setEnabled(False)
        self.ui.pushButton_cam_setting.setEnabled(True)
        self.mode = 'cam'
         
    def getResult(self,status):
        if self.ai_pause:
            return
        if status=="no img":
            return
        if not self.pause:
            status = status.split('^U^')
            result = status[1]
            score = status[2]
            self.ui.textEdit_result.setPlainText('')
            self.ui.textEdit_result.append('Predict: ' + result)
            self.ui.textEdit_result.append('Score: ' + score)   

    def start_testing_thread(self):
        self.save_config()

        if self.model_change:
            self.datasetInfo = {}
            self.cfg = config()
            self.cfg.featmap_size = (224,224)
            self.cfg.model_name = self.ui.textEdit_model_path.toPlainText()
            self.cfg.train_data_path = ""
            self.cfg.test_data_path = ""
            self.model = CNN_ResNet(self.cfg)

            self.model.moveToThread(self.thread)
            self.thread.started.connect(self.model.detect)
            self.model.sendinfo.connect(self.updateInfo)
            self.model.sendresult.connect(self.getResult)
            self.thread.start()
            self.ai_pause = False
            self.ui.pushButton_run_AI.setStyleSheet("border-image : url(./ui_img/stop_AI.png);")

            self.model_change = False
            return

        if self.ai_pause:
            self.ai_pause = False
            self.updateInfo('AI start.')
            self.ui.pushButton_run_AI.setStyleSheet("border-image : url(./ui_img/stop_AI.png);")
                
        else:
            self.ai_pause = True
            self.updateInfo('AI stop.')
            self.ui.pushButton_run_AI.setStyleSheet("border-image : url(./ui_img/run_AI.png);")

# 相機設定類別
class CamSetting(PyQt5.QtCore.QThread):
    exposure = PyQt5.QtCore.pyqtSignal(int)
    focus = PyQt5.QtCore.pyqtSignal(int)
    w = PyQt5.QtCore.pyqtSignal(int)
    h = PyQt5.QtCore.pyqtSignal(int)
    rotate = PyQt5.QtCore.pyqtSignal(int)
    flip = PyQt5.QtCore.pyqtSignal(list)
    cam_run = PyQt5.QtCore.pyqtSignal(bool)

    def __init__(self,init) -> None:
        super().__init__(None)
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow_Cam()
        self.ui.setupUi(self.MainWindow)       
        self.ui.horizontalSlider_Exposure.valueChanged.connect(self.set_exposure)
        self.ui.horizontalSlider_Focus.valueChanged.connect(self.set_focus)
        self.ui.horizontalSlider_Crop_W.valueChanged.connect(self.set_crop_w)
        self.ui.horizontalSlider_Crop_H.valueChanged.connect(self.set_crop_h)
        self.ui.radioButton_0.clicked.connect(self.set_rotate)
        self.ui.radioButton_90.clicked.connect(self.set_rotate)
        self.ui.radioButton_180.clicked.connect(self.set_rotate)
        self.ui.radioButton_270.clicked.connect(self.set_rotate)
        self.ui.checkBox_Vertical.stateChanged.connect(self.set_flip)
        self.ui.checkBox_Horizontal.stateChanged.connect(self.set_flip)
        self.ui.checkBox_Auto_Exposure.stateChanged.connect(self.auto_exposure)
        self.ui.checkBox_Auto_Focus.stateChanged.connect(self.auto_focus)
        self.ui.checkBox_11Crop.stateChanged.connect(self.equal_crop)
        self.ui.pushButton_Run.clicked.connect(self.set_cam_run)
        self.ui.pushButton_Stop.clicked.connect(self.stop)
        self.MainWindow.show()
        self.init = init
        self.get_setting(self.init)

    def set_cam_run(self):
        self.cam_run.emit(True)

    def stop(self):
        self.cam_run.emit(False)

    def get_setting(self,init):
        self.ui.textEdit_Exposure.setPlainText(str(init[0]+14))
        self.ui.horizontalSlider_Exposure.setValue(init[0]+14)
        self.ui.textEdit_Focus.setPlainText(str(init[1]))
        if init[1]==-1:
            self.ui.horizontalSlider_Focus.setEnabled(False)
        else:
            self.ui.horizontalSlider_Focus.setValue(init[1])

    def auto_exposure(self):
        if self.ui.checkBox_Auto_Exposure.isChecked():
            self.exposure.emit(9527)
        else:
            self.exposure.emit(-9527)

    def auto_focus(self):
        if self.ui.checkBox_Auto_Focus.isChecked():
            self.focus.emit(9527)
        else:
            self.focus.emit(-9527)

    def set_exposure(self,value):
        if self.ui.checkBox_Auto_Exposure.isChecked():
            return
        self.ui.textEdit_Exposure.setPlainText(str(value))
        value = value-14
        self.exposure.emit(value)

    def set_focus(self,value):
        if self.ui.checkBox_Auto_Focus.isChecked():
            return
        self.ui.textEdit_Focus.setPlainText(str(value))
        self.focus.emit(value)

    def equal_crop(self):
        if self.ui.checkBox_11Crop.isChecked():
            self.ui.horizontalSlider_Crop_H.setEnabled(False)
            self.h.emit(int(int(self.ui.textEdit_Crop_W.toPlainText())*self.init[3]/self.init[2]))
        else:
            self.ui.horizontalSlider_Crop_H.setEnabled(True)
            self.h.emit(int(self.ui.textEdit_Crop_H.toPlainText()))

    def set_crop_w(self,value):
        self.ui.textEdit_Crop_W.setPlainText(str(value))
        self.w.emit(value)
        if self.ui.checkBox_11Crop.isChecked():
            self.h.emit(int(value*self.init[3]/self.init[2]))

    def set_crop_h(self,value):
        if self.ui.checkBox_11Crop.isChecked():
            return
        self.ui.textEdit_Crop_H.setPlainText(str(value))
        self.h.emit(value)

    def set_rotate(self):
        if self.ui.radioButton_0.isChecked():
            self.rotate.emit(0)
        elif self.ui.radioButton_90.isChecked():
            self.rotate.emit(90)
        elif self.ui.radioButton_180.isChecked():
            self.rotate.emit(180)
        elif self.ui.radioButton_270.isChecked():
            self.rotate.emit(270)

    def set_flip(self):
        self.flip.emit([self.ui.checkBox_Horizontal.isChecked(),self.ui.checkBox_Vertical.isChecked()])


#app = QApplication([])
#verifier = Detector()
#verifier.MainWindow.show()
#app.exec_()