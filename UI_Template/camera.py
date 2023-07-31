from PyQt5 import QtCore
import numpy as np
import cv2
import time

class Camera(QtCore.QThread):  
    rawdata = QtCore.pyqtSignal(np.ndarray)  

    def __init__(self, parent=None ,index = 0):

        super().__init__(parent)
        self.cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if self.cam is None or not self.cam.isOpened():
            self.connect = False
            self.running = False
        else:
            self.connect = True
            self.running = False

        self.update = False
        self.width,self.height = 0,0
        self.crop_h,self.crop_w = 0,0
        self.rotate = 0
        self.flip_h,self.flip_v = False,False


    def set_exposure(self,value):
        if value==9527:
            self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # auto mode
            return
        elif value==-9527:
            self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) # # manual mode
            return
        self.cam.set(cv2.CAP_PROP_EXPOSURE,value)

    def set_focus(self,value):
        if value==9527:
            self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1) # auto mode
            return
        elif value==-9527:
            self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 0) # # manual mode
            return
        self.cam.set(cv2.CAP_PROP_FOCUS,value)

    def set_crop_w(self,value):
        self.crop_w = value

    def set_crop_h(self,value):
        self.crop_h = value

    def set_rotate(self,value):
        self.rotate = value

    def set_flip(self,hf,vf):
        self.flip_h = hf
        self.flip_v = vf

    def return_init(self):
        # CAP_PROP_EXPOSURE, CAP_PROP_FOCUS
        return [self.cam.get(cv2.CAP_PROP_EXPOSURE),self.cam.get(cv2.CAP_PROP_FOCUS),self.width,self.height]


    def run(self):
        while self.running and self.connect:
            if self.update:
                ret, img = self.cam.read()
                if self.width==0:
                   self.height,self.width,_ = img.shape 
                if ret:
                    # Crop
                    if self.crop_h!=0 and self.crop_w!=0:
                        img = img[self.crop_h:self.height-self.crop_h,self.crop_w:self.width-self.crop_w,:]
                    elif self.crop_h!=0:
                        img = img[self.crop_h:self.height-self.crop_h,:,:]
                    elif self.crop_w!=0:
                        img = img[:,self.crop_w:self.width-self.crop_w,:]
                    # Rotate
                    if self.rotate==90:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif self.rotate==180:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                    elif self.rotate==270:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    # Flip
                    if self.flip_h and self.flip_v:
                        img = cv2.flip(img, -1)
                    elif self.flip_h:
                        img = cv2.flip(img, 1)
                    elif self.flip_v:
                        img = cv2.flip(img, 0)
                    self.rawdata.emit(img)
                else:    
                    print("Warning!!!")
                    self.connect = False
            else:
                time.sleep(1)

    def set_update(self, update):
        self.update = update

    def open(self):
        if self.connect:
            self.running = True  

    def stop(self):
        if self.connect:
            self.running = False   

    def close(self):

        if self.connect:
            self.running = False    
            time.sleep(1)
            self.cam.release()      