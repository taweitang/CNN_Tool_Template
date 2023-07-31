import torch
import torch.nn as nn
import torch.nn.functional as F
#from extractors.feature import Extractor
from models.skipDFR.feature import Extractor
from torch.utils.data import DataLoader
import torch.optim as optim
#from data.MVTec import NormalDataset, TrainTestDataset

import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage import measure
from skimage.transform import resize
import pandas as pd
import cv2
import shutil

from models.skipDFR.feat_cae import FeatCAEUNET

import joblib
from sklearn.decomposition import PCA

from models.multiDFR.utils import *
from models.skipDFR.utils import *

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from zipfile import ZipFile
from scipy.ndimage import gaussian_filter
import torchvision.transforms as transforms
import statistics
from PIL import Image


## Davud statistic
def histogram(y_test,score,save_path):
    goods = []
    bads = []
    for i in range(len(y_test)):
        if(y_test[i]==0):
            goods.append(score[i])
        else:
            bads.append(score[i])
    std_good = round(np.std(goods),4)
    mean_good = round(sum(goods) / len(goods),2)
    std_bad = round(np.std(bads),4)
    mean_bad = round(sum(bads) / len(bads),2)
    bins = np.linspace(int(min(score))-1, int(max(score))+1, int(max(score))-int(min(score))+3)
    plt.figure()
    plt.hist(goods, bins, alpha = 0.5, label='good, mean = '+ str(mean_good) + ', std = ' + str(std_good) + ', max = ' + str(round(max(goods),2)) + ', min = ' + str(round(min(goods),2)))
    plt.hist(bads, bins, alpha = 0.5, label='bad, mean = '+ str(mean_bad) + ', std = ' + str(std_bad) + ', max = ' + str(round(max(bads),2)) + ', min = ' + str(round(min(bads),2)))
    plt.legend(loc='upper left')
    plt.savefig(save_path)

 # Functions
def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    

def cvt2heatmap(gray,threshold = 0.45):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    mask = np.where(gray >= threshold*255, 255, 0).astype(np.uint8)
    heatmap_masked = cv2.bitwise_and(heatmap, heatmap, mask=mask)

    return heatmap_masked,mask

class SkipDFR(QObject):

    # Callback function
    finished = pyqtSignal()
    sendinfo = pyqtSignal(str)
    progress = pyqtSignal(str)
    sendresult = pyqtSignal(str)
    """
    Anomaly segmentation model: DFR.
    """
    def __init__(self, cfg):
        super(SkipDFR, self).__init__()
        self.cfg = cfg
        self.path = cfg.save_path    # model and results saving path

        self.n_layers = len(cfg.cnn_layers)
        self.n_dim = cfg.latent_dim

        self.log_step = 10
        self.data_name = cfg.data_name

        self.img_size = cfg.img_size
        self.threshold = cfg.thred
        self.device = torch.device(cfg.device)
        self.forced_stop = False
        self.detect_imgs = []

        # feature extractor
        self.extractor = Extractor(backbone=cfg.backbone,
                 cnn_layers=cfg.cnn_layers,
                 upsample=cfg.upsample,
                 is_agg=cfg.is_agg,
                 kernel_size=cfg.kernel_size,
                 stride=cfg.stride,
                 dilation=cfg.dilation,
                 featmap_size=cfg.featmap_size,
                 device=cfg.device).to(self.device)

        # datasest
        self.train_data_path = cfg.train_data_path
        self.test_data_path = cfg.test_data_path
        self.train_data = self.build_dataset(is_train=True)
        self.test_data = self.build_dataset(is_train=False)

        # dataloader
        self.test_data_loader = DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=0)
        if self.train_data and self.train_data!="":
            self.train_data_loader = DataLoader(self.train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
            self.eval_data_loader = DataLoader(self.train_data, batch_size=10, shuffle=False, num_workers=0)
        


        # autoencoder classifier
        if self.train_data and self.train_data!="":
            self.autoencoder, self.model_name = self.build_classifier()
            if cfg.model_name != "":
                self.model_name = cfg.model_name
            #print("model name:", self.model_name)

            # optimizer
            self.lr = cfg.lr
            self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.lr, weight_decay=0)

            #0821
            # saving paths
            self.model_path = self.path
            #self.subpath = self.data_name + "/" + self.model_name
            #self.model_path = os.path.join(self.path, "models/" + self.subpath + "/model")
            #if not os.path.exists(self.model_path):
            #    os.makedirs(self.model_path)
            #self.eval_path = os.path.join(self.path, "models/" + self.subpath + "/eval")
            #if not os.path.exists(self.eval_path):
            #    os.makedirs(self.eval_path)
        else:
            self.autoencoder, _ = self.build_classifier()

    def build_classifier(self):
        #self.load_dim(self.model_path)
        #之後要拿掉
        self.n_dim,in_feat = 231,1792

        if os.path.isfile('./temp/config.ini'):
            f = open("./temp/config.ini", "r")
            text = f.readline()
            text = text.split(',')
            self.n_dim,in_feat = int(text[1]),int(text[2])
            f.close()

        if self.n_dim is None:
            print("Estimating one class classifier AE parameter...")
            self.sendinfo.emit("Estimating one class classifier AE parameter...")
            feats = torch.Tensor()
            for i, normal_img in enumerate(self.eval_data_loader):
                i += 1
                if i > 1:
                    break
                normal_img = normal_img.to(self.device)
                feat = self.extractor.feat_vec(normal_img)
                feats = torch.cat([feats, feat.cpu()], dim=0)
            # to numpy
            feats = feats.detach().numpy()
            # estimate parameters for mlp
            pca = PCA(n_components=0.90)    # 0.9 here try 0.8
            pca.fit(feats)
            n_dim, in_feat = pca.components_.shape
            print("AE Parameter (in_feat, n_dim): ({}, {})".format(in_feat, n_dim))
            self.n_dim = n_dim
        elif in_feat is None:
            for i, normal_img in enumerate(self.eval_data_loader):
                i += 1
                if i > 1:
                    break
                normal_img = normal_img.to(self.device)
                feat = self.extractor.feat_vec(normal_img)
            in_feat = feat.shape[1]
        
        self.in_feat = in_feat
        print("BN?:", self.cfg.is_bn)
        autoencoder = FeatCAEUNET(in_channels=in_feat, latent_dim=self.n_dim, is_bn=self.cfg.is_bn).to(self.device)
        model_name = "AnoSegDFR({})_{}_l{}_d{}_s{}_k{}_{}".format('BN' if self.cfg.is_bn else 'noBN',
                                                                self.cfg.backbone, self.n_layers,
                                                                self.n_dim, self.cfg.stride[0],
                                                                self.cfg.kernel_size[0], self.cfg.upsample)

        return autoencoder, model_name


    def build_dataset(self, is_train,force_norm = False):
        from models.multiDFR.MVTec import NormalDataset, TestDataset
        normal_data_path = self.train_data_path
        abnormal_data_path = self.test_data_path
        if force_norm:
            dataset = NormalDataset(abnormal_data_path + "good/", normalize=True)
        else:
            if is_train:
                dataset = NormalDataset(normal_data_path, normalize=True)
            else:
                dataset = TestDataset(path=abnormal_data_path)
        return dataset

    def train_statistics(self):
        max_score = 0
        min_score = 1000000000
        paths_temp = []
        for root, dirs, files in os.walk(self.test_data_path):
            for f in files:
                fullpath = os.path.join(root, f)
                paths_temp.append(fullpath)
        
        for f in paths_temp:
            if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp'):
                name = os.path.join(self.test_data_path,f)
                resize = transforms.Resize(size=(256, 256), interpolation=Image.NEAREST)
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
                img = transform(Image.open(name)).float().to(self.device)
                img = img[None,:]
                temp_results = self.score_and_map(img)
                score = temp_results[0].data.cpu().numpy()
                if score>max_score:
                    max_score = score
                if score<min_score:
                    min_score = score
        self.min = min_score
        self.max = max_score

    def train(self):
        self.sendinfo.emit('Start training (Epoch = {}, lr = {})'.format(self.cfg.epochs, self.cfg.lr))
        #if self.load_model():
        #    print("Model Loaded.")
        #    return

        start_time = time.time()

        torch.backends.cudnn.benchmark = True
        # train
        iters_per_epoch = len(self.train_data_loader)  # total iterations every epoch
        epochs = self.cfg.epochs  # total epochs
        self.progress.emit(str(epochs) +',' +str(iters_per_epoch*epochs) + ',*')
        best_img_auc = 0
        best_pixel_auc = 0
        best_img_epoch = 0
        best_pixel_epoch = 0
        biAUCTD,biAUCBS,biAUCTDBS = 0,0,0
        bpAUCTD,bpAUCBS,bpAUCTDBS = 0,0,0
        biTDROC = []
        piTDROC = []
        for epoch in range(1, epochs+1):
            self.extractor.train()
            self.autoencoder.train()
            losses = []
            for i, normal_img in enumerate(self.train_data_loader):
                normal_img = normal_img.to(self.device)
                # forward and backward
                total_loss = self.optimize_step(normal_img)

                # statistics and logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                
                # tracking loss
                losses.append(loss['total_loss'])
                #[epoch,step,loss,auc]
                self.progress.emit(str(epoch)+','+str((epoch-1)*iters_per_epoch+i+1)+','+str(loss['total_loss'])+',0')
            
            if epoch % 1 == 0:

                print('Epoch {}/{}'.format(epoch, epochs))
                print('-' * 10)
                elapsed = time.time() - start_time
                total_time = ((epochs * iters_per_epoch) - (epoch * iters_per_epoch + i)) * elapsed / (
                        epoch * iters_per_epoch + i + 1)
                epoch_time = (iters_per_epoch - i) * elapsed / (epoch * iters_per_epoch + i + 1)

                epoch_time = str(datetime.timedelta(seconds=epoch_time))
                total_time = str(datetime.timedelta(seconds=total_time))
                elapsed = str(datetime.timedelta(seconds=elapsed))

                log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}]".format(
                    elapsed, epoch_time, total_time, epoch, epochs, i + 1, iters_per_epoch)

                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                    self.sendinfo.emit('Epoch {}/{}'.format(epoch, epochs) + ", {}: {:.4f}".format(tag, value))
                print(log)
                
                

            if epoch % 5 == 0 or self.forced_stop:
                # save model
                self.save_model()
                img_auc,iAUCTD,iAUCBS,iAUCTDBS,iTDROC = self.validation(epoch)
                #pixel_auc,pAUCTD,pAUCBS,pAUCTDBS,pTDROC = self.validation(epoch,usemask =True)
                if (img_auc>best_img_auc):
                    best_img_auc = img_auc
                    best_img_epoch = epoch
                    biAUCTD = iAUCTD
                    biAUCBS = iAUCBS
                    biAUCTDBS = iAUCTDBS
                    biTDROC = iTDROC
                '''if (pixel_auc>best_pixel_auc):
                    best_pixel_auc = pixel_auc
                    best_pixel_epoch = epoch
                    bpAUCTD = pAUCTD
                    bpAUCBS = pAUCBS
                    bpAUCTDBS = pAUCTDBS
                    bpTDROC = pTDROC'''

                self.progress.emit('0,' + str(epoch*iters_per_epoch)+ ',0,' + str(img_auc))
#             print("Cost total time {}s".format(time.time() - start_time))
#             print("Done.")
            #self.tracking_loss(epoch, np.mean(np.array(losses)))

            if self.forced_stop:
                break

        # save model
        self.save_model()
        #with open(self.path + "/" + self.data_name + "_img_auc" + str(round(best_img_auc,3)) + ", " + str(round(biAUCTD,3))+ ", " + str(round(biAUCBS,3))+ ", " + str(round(biAUCTDBS,3)), 'w') as file:
        #    file.write('best img epoch = ' + str(best_img_epoch) + ',best img auc = ' + str(best_img_auc))
        #with open(self.path + "/" + self.data_name + "_pixel_auc" + str(round(best_pixel_auc,3)) + ", " + str(round(bpAUCTD,3))+ ", " + str(round(bpAUCBS,3))+ ", " + str(round(bpAUCTDBS,3)), 'w') as file:
        #    file.write('best img epoch = ' + str(best_pixel_epoch) + ',best img auc = ' + str(best_pixel_auc))
        print("Cost total time {}s".format(time.time() - start_time))
        print("Done.")
        self.sendinfo.emit("Done.")

        if self.forced_stop:
            self.forced_stop = False
            self.sendinfo.emit('Training process have been stoped.')
            
        # relese gpu
        torch.cuda.empty_cache()

    def tracking_loss(self, epoch, loss):
        out_file = os.path.join(self.eval_path, '{}_epoch_loss.csv'.format(self.model_name))
        if not os.path.exists(out_file):
            with open(out_file, mode='w') as f:
                f.write("Epoch" + ",loss" + "\n")
        with open(out_file, mode='a+') as f:
            f.write(str(epoch) + "," + str(loss) + "\n")

    def optimize_step(self, input_data):
        self.extractor.train()
        self.autoencoder.train()

        self.optimizer.zero_grad()

        # forward
        input_data = self.extractor(input_data)

        # print(input_data.size())
        dec = self.autoencoder(input_data)

        # loss
        #total_loss = self.autoencoder.loss_function(dec[0], dec[1])
        total_loss = self.autoencoder.loss_function(dec, input_data.detach().data)

        # self.reset_grad()
        total_loss.backward()

        self.optimizer.step()

        return total_loss

    def score(self, input):
        """
        Args:
            input: image with size of (img_size_h, img_size_w, channels)
        Returns:
            score map with shape (img_size_h, img_size_w)
        """
        self.extractor.eval()
        self.autoencoder.eval()

        input = self.extractor(input)
        dec = self.autoencoder(input)

        # sample energy
        scores = self.autoencoder.compute_energy(dec, input.detach().data)
        scores = scores.reshape((1, 1, self.extractor.out_size[0], self.extractor.out_size[1]))    # test batch size is 1.
        scores = nn.functional.interpolate(scores, size=self.img_size, mode="bilinear", align_corners=True).squeeze()
        # print("score shape:", scores.shape)
        return scores

    def score_img_scale(self, input):
        self.extractor.eval()
        self.autoencoder.eval()
        input = self.extractor(input)
        dec = self.autoencoder(input)
        #score = self.autoencoder.loss_function(dec[0], dec[1])
        score = self.autoencoder.loss_function(dec, input.detach().data)
        return score
    
    def score_and_map(self,input,in_padding=15):
        self.extractor.eval()
        self.autoencoder.eval()
        input = self.extractor(input)
        dec = self.autoencoder(input)
        scores = self.autoencoder.compute_energy(dec, input.detach().data)
        scores = scores.reshape((1, 1, self.extractor.out_size[0], self.extractor.out_size[1]))    # test batch size is 1.
        map = nn.functional.interpolate(scores, size=self.img_size, mode="bilinear", align_corners=True).squeeze()

        # 消除邊緣雜訊
        map[:in_padding, :] = 0
        map[-in_padding:, :] = 0
        map[:, :in_padding] = 0
        map[:, -in_padding:] = 0

        #score = self.autoencoder.loss_function(dec, input.detach().data)
        score = torch.mean(map)
        return score,map

    def segment(self, input, threshold=0.5):
        """
        Args:
            input: image with size of (img_size_h, img_size_w, channels)
        Returns:
            score map and binary score map with shape (img_size_h, img_size_w)
        """
        # predict
        scores = self.score(input).data.cpu().numpy()

        # binary score
        print("threshold:", threshold)
        binary_scores = np.zeros_like(scores)    # torch.zeros_like(scores)
        binary_scores[scores <= threshold] = 0
        binary_scores[scores > threshold] = 1

        return scores, binary_scores

    def segment_evaluation(self):
        i = 0
        metrics = []
        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)
            scores, binary_scores = self.segment(img, threshold=self.threshold)

            # show something
            #     plt.figure()
            #     ax1 = plt.subplot(1, 2, 1)
            #     ax1.imshow(resize(mask[0], (256, 256)))
            #     ax1.set_title("gt")

            #     ax2 = plt.subplot(1, 2, 2)
            #     ax2.imshow(scores)
            #     ax2.set_title("pred")

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            self.save_seg_results(normalize(scores), binary_scores, mask, name)
            # metrics of one batch
            if name.split("/")[-2] != "good":
                specificity, sensitivity, accuracy, coverage, auc = spec_sensi_acc_iou_auc(mask, binary_scores, scores)
                metrics.append([specificity, sensitivity, accuracy, coverage, auc])
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))
        # metrics over all data
        metrics = np.array(metrics)
        metrics_mean = metrics.mean(axis=0)
        metrics_std = metrics.std(axis=0)
        print("metrics: specificity, sensitivity, accuracy, iou, auc")
        print("mean:", metrics_mean)
        print("std:", metrics_std)
        print("threshold:", self.threshold)

    def save_paths(self):
        # generating saving paths
        score_map_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/score_map")
        if not os.path.exists(score_map_path):
            os.makedirs(score_map_path)

        binary_score_map_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/binary_score_map")
        if not os.path.exists(binary_score_map_path):
            os.makedirs(binary_score_map_path)

        gt_pred_map_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/gt_pred_score_map")
        if not os.path.exists(gt_pred_map_path):
            os.makedirs(gt_pred_map_path)

        mask_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/mask")
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        gt_pred_seg_image_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/gt_pred_seg_image")
        if not os.path.exists(gt_pred_seg_image_path):
            os.makedirs(gt_pred_seg_image_path)

        return score_map_path, binary_score_map_path, gt_pred_map_path, mask_path, gt_pred_seg_image_path

    def save_seg_results(self, scores, binary_scores, mask, name):
        score_map_path, binary_score_map_path, gt_pred_score_map, mask_path, gt_pred_seg_image_path = self.save_paths()
        img_name = name.split("/")
        img_name = "-".join(img_name[-2:])
        print(img_name)
        # score map
        imsave(os.path.join(score_map_path, "{}".format(img_name)), scores)

        # binary score map
        imsave(os.path.join(binary_score_map_path, "{}".format(img_name)), binary_scores)

        # mask
        imsave(os.path.join(mask_path, "{}".format(img_name)), mask)

        # # pred vs gt map
        # imsave(os.path.join(gt_pred_score_map, "{}".format(img_name)), normalize(binary_scores + mask))
        visulization_score(img_file=name, mask_path=mask_path,
                     score_map_path=score_map_path, saving_path=gt_pred_score_map)
        # pred vs gt image
        visulization(img_file=name, mask_path=mask_path,
                     score_map_path=binary_score_map_path, saving_path=gt_pred_seg_image_path)

    def save_model(self, epoch=0):
        self.train_statistics()
        # save model weights
        ae_path = os.path.join(self.model_path, 'autoencoder.pth')
        ini_path = os.path.join(self.model_path, 'config.ini')
        #ae_path = './autoencoder.pth'
        #ini_path = './config.ini'
        torch.save({'autoencoder': self.autoencoder.state_dict()}, ae_path)
        with open(os.path.join(self.model_path, 'config.ini'), 'w') as f:
            # model index, self.n_dim, self.in_feat
            f.write('{0},{1},{2},{3},{4}'.format(str(self.cfg.model_index),str(self.n_dim),str(self.in_feat),str(self.min),str(self.max)))
            #f.write('{0},{1},{2}'.format(str(self.cfg.model_index),str(self.n_dim),str(self.in_feat)))
        with ZipFile(os.path.join(self.model_path, 'AD_model.pbz'),'w') as zip:
            # writing each file one by one
            zip.write(ae_path,'autoencoder.pth')
            zip.write(ini_path,'config.ini')
        os.remove(ae_path)
        os.remove(ini_path)
        #np.save(os.path.join(self.model_path, 'n_dim.npy'), self.n_dim)

    def load_model(self, path=None):
        print("Loading model...")
        if path is None:
            model_path = os.path.join(self.model_path, 'autoencoder.pth')
        else:
            model_path = path
        print("model path:", model_path)
        if not os.path.exists(model_path):
            print("Model not exists.")
            return False

        if torch.cuda.is_available():
            data = torch.load(model_path)
        else:
            data = torch.load(model_path,
                                map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU, using a function

        self.autoencoder.load_state_dict(data['autoencoder'])
        print("Model loaded:", model_path)
        return True

    # def save_dim(self):
    #     np.save(os.path.join(self.model_path, 'n_dim.npy'))

    def load_dim(self, model_path):
        dim_path = os.path.join(model_path, 'n_dim.npy')
        if not os.path.exists(dim_path):
            print("Dim not exists.")
            self.n_dim = None
        else:
            self.n_dim = np.load(os.path.join(model_path, 'n_dim.npy'))

    ########################################################
    #  Evaluation (testing)
    ########################################################
    def segmentation_results(self):
        def normalize(x):
            return x/x.max()

        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)
            scores, binary_scores = self.segment(img, threshold=self.threshold)

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            if name[0].split("/")[-2] != "good":
                self.save_seg_results(normalize(scores), binary_scores, mask, name)
            # self.save_seg_results((scores-score_min)/score_range, binary_scores, mask, name)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))

    ######################################################
    #  Evaluation of segmentation
    ######################################################
    def save_segment_paths(self, fpr):
        # generating saving paths
        binary_score_map_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/fpr_{}/binary_score_map".format(fpr))
        if not os.path.exists(binary_score_map_path):
            os.makedirs(binary_score_map_path)

        mask_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/fpr_{}/mask".format(fpr))
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        gt_pred_seg_image_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/fpr_{}/gt_pred_seg_image".format(fpr))
        if not os.path.exists(gt_pred_seg_image_path):
            os.makedirs(gt_pred_seg_image_path)

        return binary_score_map_path, mask_path, gt_pred_seg_image_path

    def save_segment_results(self, binary_scores, mask, name, fpr):
        binary_score_map_path, mask_path, gt_pred_seg_image_path = self.save_segment_paths(fpr)
        img_name = name.split("/")
        img_name = "-".join(img_name[-2:])
        print(img_name)
        # binary score map
        imsave(os.path.join(binary_score_map_path, "{}".format(img_name)), binary_scores)

        # mask
        imsave(os.path.join(mask_path, "{}".format(img_name)), mask)

        # pred vs gt image
        visulization(img_file=name, mask_path=mask_path,
                     score_map_path=binary_score_map_path, saving_path=gt_pred_seg_image_path)

    def estimate_thred_with_fpr(self, expect_fpr=0.05):
        """
        Use training set to estimate the threshold.
        """
        threshold = 0
        scores_list = []
        for i, normal_img in enumerate(self.train_data_loader):
            normal_img = normal_img[0:1].to(self.device)
            scores_list.append(self.score(normal_img).data.cpu().numpy())
        scores = np.concatenate(scores_list, axis=0)

        # find the optimal threshold
        max_step = 100
        min_th = scores.min()
        max_th = scores.max()
        delta = (max_th - min_th) / max_step
        for step in range(max_step):
            threshold = max_th - step * delta
            # segmentation
            binary_score_maps = np.zeros_like(scores)
            binary_score_maps[scores <= threshold] = 0
            binary_score_maps[scores > threshold] = 1

            # estimate the optimal threshold base on user defined min_area
            fpr = binary_score_maps.sum() / binary_score_maps.size
            print(
                "threshold {}: find fpr {} / user defined fpr {}".format(threshold, fpr, expect_fpr))
            if fpr >= expect_fpr:  # find the optimal threshold
                print("find optimal threshold:", threshold)
                print("Done.\n")
                break
        return threshold

    def segment_evaluation_with_fpr(self, expect_fpr=0.05):
        # estimate threshold
        thred = self.estimate_thred_with_fpr(expect_fpr=expect_fpr)

        # segment
        i = 0
        metrics = []
        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)
            scores, binary_scores = self.segment(img, threshold=thred)

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            self.save_segment_results(binary_scores, mask, name, expect_fpr)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))
        print("threshold:", thred)

    def segment_evaluation_with_otsu_li(self, seg_method='otsu'):
        """
        ref: skimage.filters.threshold_otsu
        skimage.filters.threshold_li
        e.g.
        thresh = filters.threshold_otsu(image) #返回一个阈值
        dst =(image <= thresh)*1.0 #根据阈值进行分割
        """
        from skimage.filters import threshold_li
        from skimage.filters import threshold_otsu

        # segment
        thred = 0
        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)

            # estimate threshold and seg
            if seg_method == 'otsu':
                thred = threshold_otsu(img.detach().cpu().numpy())
            else:
                thred = threshold_li(img.detach().cpu().numpy())
            scores, binary_scores = self.segment(img, threshold=thred)

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            self.save_segment_results(binary_scores, mask, name, seg_method)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))
        print("threshold:", thred)

    def segmentation_evaluation(self):
        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return
        self.segment_evaluation_with_fpr(expect_fpr=self.cfg.except_fpr)


    def validation(self, epoch, usemask =False):
        from sklearn.metrics import roc_auc_score
        from sklearn import preprocessing
        i = 0
        time_start = time.time()   
        if usemask:
            scores = []
            masks = []
            for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
                i += 1
                # data
                img = img.to(self.device)
                mask = mask.squeeze().numpy()

                # score
                score = self.score(img).data.cpu().numpy()

                masks.append(mask)
                scores.append(score)
                #print("Batch {},".format(i), "Cost total time {}s".format(time.time() - time_start))

            # as array
            masks = np.array(masks)
            masks[masks <= 0.5] = 0
            masks[masks > 0.5] = 1
            masks = masks.astype(np.bool)
            scores = np.array(scores)

            # auc score
            auc_score, roc = auc_roc(masks, scores)
            AUC,AUCTD,AUCBS,AUCTDBS,TDROC = AUCs(masks,scores,True)
            # metrics over all data
            print("pixel_auc:", auc_score)
            print("AUC:",AUC,", AUCTD:",AUCTD,", AUCBS:",AUCBS,", AUCTDBS:",AUCTDBS)
            
            return AUC,AUCTD,AUCBS,AUCTDBS,TDROC
        else:
            scores = []
            gt = []
            for i, (img,_,name) in enumerate(self.test_data_loader):  # batch size is 1.
                i += 1
                # data
                img = img.to(self.device)
                # score
                #print('val img:', img)
                score = self.score_img_scale(img).data.cpu().numpy()
                scores.append(score)
                #print("Batch {},".format(i), "Cost total time {}s".format(time.time() - time_start))
                #print('name: ',name)
                # GT
                if "good" in name[0]:
                    gt.append(0)
                else:
                    gt.append(1)
                
            # auc score
            #recalls,precisions,TPRS,FPRS = statistics(gt, scores)
            #histogram(self.gt_list_img_lvl,self.pred_list_img_lvl,os.path.join(self.sample_path, 'histogram.png'))
            auc_score = roc_auc_score(gt,scores)
            print("image_auc:", auc_score)
            #scores = np.array(scores).reshape(-1,1)
            #scaler = preprocessing.MinMaxScaler()
            #scores = scaler.fit_transform(scores)
            AUC,AUCTD,AUCBS,AUCTDBS,TDROC = AUCs(gt,scores)
            #auc_score, roc = auc_roc(masks, scores)
            # metrics over all data
            print("AUC:",AUC,", AUCTD:",AUCTD,", AUCBS:",AUCBS,", AUCTDBS:",AUCTDBS)

            
            return AUC,AUCTD,AUCBS,AUCTDBS,TDROC

    def detect(self):
        if self.load_model(self.cfg.model_name):
            print("Model Loaded.")
            self.sendinfo.emit("Model Loaded.")
        else:
            print("None pretrained models.")
            return
    
        shutil.rmtree('./heatmapTemp/')
        os.mkdir('./heatmapTemp/')

        self.sendinfo.emit("Start to detect.")
        count = 0
        while True:
            if len(self.detect_imgs)==0:
                self.sendresult.emit("no img")
                time.sleep(0.001)
                continue
            else:
                image = self.detect_imgs.pop(0)
                ori_size = image.shape
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image)
                resize = transforms.Resize(size=(256, 256), interpolation=Image.NEAREST)
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
                img = transform(img).float().to(self.device)
                img = img[None,:]
                temp_results = self.score_and_map(img)
                score,anomaly_map = temp_results[0].data.cpu().numpy(),temp_results[1].data.cpu().numpy()
                anomaly_map_blur = gaussian_filter(anomaly_map, sigma=4)
                anomaly_map_norm = min_max_norm(anomaly_map_blur)
                heatmap,mask = cvt2heatmap(anomaly_map_norm * 255)
                heatmap = cv2.resize(heatmap_on_image(heatmap,image),[ori_size[1],ori_size[0]])
                heatmap_path = str(count) + "_ht.png"
                heatmap_path = './heatmapTemp/' + heatmap_path
                cv2.imwrite(heatmap_path,heatmap)
                self.sendresult.emit("{0}^U^{1}^U^{2}".format('None',heatmap_path.replace('\\','/'),str(score)))
                count+=1

    def verify(self):
        if self.load_model(self.cfg.model_name):
            print("Model Loaded.")
            self.sendinfo.emit("Model Loaded.")
        else:
            print("None pretrained models.")
            return
        
        shutil.rmtree('./heatmapTemp/')
        os.mkdir('./heatmapTemp/')

        self.sendinfo.emit("Start to verify.")
        max_score = 0
        min_score = 1000000000
        self.sendresult.emit("{0}^U^{1}^U^{2}".format('init','',str(len(self.test_data_loader))))
        paths_temp = []
        for root, dirs, files in os.walk(self.test_data_path):
            for f in files:
                fullpath = os.path.join(root, f)
                paths_temp.append(fullpath)
        
        for f in paths_temp:
            if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp'):
                name = os.path.join(self.test_data_path,f)
                resize = transforms.Resize(size=(256, 256), interpolation=Image.NEAREST)
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
                img = transform(Image.open(name)).float().to(self.device)
                img = img[None,:]
                temp_results = self.score_and_map(img)
                score,anomaly_map = temp_results[0].data.cpu().numpy(),temp_results[1].data.cpu().numpy()
                anomaly_map_blur = gaussian_filter(anomaly_map, sigma=4)
                anomaly_map_norm = min_max_norm(anomaly_map_blur)
                heatmap,mask = cvt2heatmap(anomaly_map_norm * 255)
                image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
                ori_size = image.shape
                heatmap = cv2.resize(heatmap_on_image(heatmap,image),[ori_size[1],ori_size[0]])
                mask = cv2.resize(mask,[ori_size[1],ori_size[0]])
                heatmap_path = name.replace(".png","_ht.png").replace(".jpg","_ht.png")
                heatmap_path = heatmap_path.split('\\')[-2] + heatmap_path.split('\\')[-1]
                heatmap_path = './heatmapTemp/' + heatmap_path
                cv2.imwrite(heatmap_path,heatmap)
                mask_path = heatmap_path.replace("_ht.png",".m.png")
                cv2.imwrite(mask_path,mask)
                if score>max_score:
                    max_score = score
                if score<min_score:
                    min_score = score
                # image_path,heatmap_path,score
                self.sendresult.emit("{0}^U^{1}^U^{2}".format(name.replace('\\','/'),heatmap_path.replace('\\','/'),str(score)))
        self.sendresult.emit("{0}^U^{1}^U^{2}".format('max_min',str(max_score),str(min_score)))
        self.sendinfo.emit("Task done.")



    def metrics_evaluation(self, expect_fpr=0.3, max_step=5000):
        from sklearn.metrics import auc
        from sklearn.metrics import roc_auc_score, average_precision_score
        from skimage import measure
        import pandas as pd

        if self.load_model(self.cfg.model_name):
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return

        print("Calculating AUC, IOU, PRO metrics on testing data...")
        masks = []
        scores = []
        for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            # data
            img = img.to(self.device)
            mask = mask.squeeze().numpy()

            # anomaly score
            # anomaly_map = self.score(img).data.cpu().numpy()
            anomaly_map = self.score(img).data.cpu().numpy()

            masks.append(mask)
            scores.append(anomaly_map)
            #print("Batch {},".format(i), "Cost total time {}s".format(time.time() - time_start))

        # as array
        masks = np.array(masks)
        scores = np.array(scores)
        
        # binary masks
        masks[masks <= 0.5] = 0
        masks[masks > 0.5] = 1
        masks = masks.astype(np.bool)
        
        # auc score (image level) for detection
        labels = masks.any(axis=1).any(axis=1)
#         preds = scores.mean(1).mean(1)
        preds = scores.max(1).max(1)    # for detection
        det_auc_score = roc_auc_score(labels, preds)
        det_pr_score = average_precision_score(labels, preds)
        
        # auc score (per pixel level) for segmentation
        seg_auc_score = roc_auc_score(masks.ravel(), scores.ravel())
        seg_pr_score = average_precision_score(masks.ravel(), scores.ravel())
        # metrics over all data
        print(f"Det AUC: {det_auc_score:.4f}, Seg AUC: {seg_auc_score:.4f}")
        print(f"Det PR: {det_pr_score:.4f}, Seg PR: {seg_pr_score:.4f}")
        
        # per region overlap and per image iou
        max_th = scores.max()
        min_th = scores.min()
        delta = (max_th - min_th) / max_step
        
        ious_mean = []
        ious_std = []
        pros_mean = []
        pros_std = []
        threds = []
        fprs = []
        binary_score_maps = np.zeros_like(scores, dtype=np.bool)
        for step in range(max_step):
            thred = max_th - step * delta
            # segmentation
            binary_score_maps[scores <= thred] = 0
            binary_score_maps[scores > thred] = 1

            pro = []    # per region overlap
            iou = []    # per image iou
            # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
            # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
            for i in range(len(binary_score_maps)):    # for i th image
                # pro (per region level)
                label_map = measure.label(masks[i], connectivity=2)
                props = measure.regionprops(label_map)
                for prop in props:
                    x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
                    cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                    # cropped_mask = masks[i][x_min:x_max, y_min:y_max]   # bug!
                    cropped_mask = prop.filled_image    # corrected!
                    intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                    pro.append(intersection / prop.area)
                # iou (per image level)
                intersection = np.logical_and(binary_score_maps[i], masks[i]).astype(np.float32).sum()
                union = np.logical_or(binary_score_maps[i], masks[i]).astype(np.float32).sum()
                if masks[i].any() > 0:    # when the gt have no anomaly pixels, skip it
                    iou.append(intersection / union)
            # against steps and average metrics on the testing data
            ious_mean.append(np.array(iou).mean())
#             print("per image mean iou:", np.array(iou).mean())
            ious_std.append(np.array(iou).std())
            pros_mean.append(np.array(pro).mean())
            pros_std.append(np.array(pro).std())
            # fpr for pro-auc
            masks_neg = ~masks
            fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
            fprs.append(fpr)
            threds.append(thred)
            
        # as array
        threds = np.array(threds)
        pros_mean = np.array(pros_mean)
        pros_std = np.array(pros_std)
        fprs = np.array(fprs)
        
        ious_mean = np.array(ious_mean)
        ious_std = np.array(ious_std)
        
        # save results
        data = np.vstack([threds, fprs, pros_mean, pros_std, ious_mean, ious_std])
        df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
                                                        'pros_mean', 'pros_std',
                                                        'ious_mean', 'ious_std'])
        # save results
        df_metrics.to_csv(os.path.join(self.eval_path, 'thred_fpr_pro_iou.csv'), sep=',', index=False)

        
        # best per image iou
        best_miou = ious_mean.max()
        print(f"Best IOU: {best_miou:.4f}")
        
        # default 30% fpr vs pro, pro_auc
        idx = fprs <= expect_fpr    # find the indexs of fprs that is less than expect_fpr (default 0.3)
        fprs_selected = fprs[idx]
        fprs_selected = rescale(fprs_selected)    # rescale fpr [0,0.3] -> [0, 1]
        pros_mean_selected = pros_mean[idx]    
        pro_auc_score = auc(fprs_selected, pros_mean_selected)
        print("pro auc ({}% FPR):".format(int(expect_fpr*100)), pro_auc_score)

        # save results
        data = np.vstack([threds[idx], fprs[idx], pros_mean[idx], pros_std[idx]])
        df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
                                                        'pros_mean', 'pros_std'])
        df_metrics.to_csv(os.path.join(self.eval_path, 'thred_fpr_pro_{}.csv'.format(expect_fpr)), sep=',', index=False)

        # save auc, pro as 30 fpr
        with open(os.path.join(self.eval_path, 'pr_auc_pro_iou_{}.csv'.format(expect_fpr)), mode='w') as f:
                f.write("det_pr, det_auc, seg_pr, seg_auc, seg_pro, seg_iou\n")
                f.write(f"{det_pr_score:.5f},{det_auc_score:.5f},{seg_pr_score:.5f},{seg_auc_score:.5f},{pro_auc_score:.5f},{best_miou:.5f}")    
            

    def metrics_detecion(self, expect_fpr=0.3, max_step=5000):
        from sklearn.metrics import auc
        from sklearn.metrics import roc_auc_score, average_precision_score
        from skimage import measure
        import pandas as pd

        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return

        print("Calculating AUC, IOU, PRO metrics on testing data...")
        time_start = time.time()
        masks = []
        scores = []
        for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            # data
            img = img.to(self.device)
            mask = mask.squeeze().numpy()

            # anomaly score
            # anomaly_map = self.score(img).data.cpu().numpy()
            anomaly_map = self.score(img).data.cpu().numpy()

            masks.append(mask)
            scores.append(anomaly_map)
            #print("Batch {},".format(i), "Cost total time {}s".format(time.time() - time_start))

        # as array
        masks = np.array(masks)
        scores = np.array(scores)
        
        # binary masks
        masks[masks <= 0.5] = 0
        masks[masks > 0.5] = 1
        masks = masks.astype(np.bool)
        
        # auc score (image level) for detection
        labels = masks.any(axis=1).any(axis=1)
#         preds = scores.mean(1).mean(1)
        preds = scores.max(1).max(1)    # for detection
        det_auc_score = roc_auc_score(labels, preds)
        det_pr_score = average_precision_score(labels, preds)
        
        # auc score (per pixel level) for segmentation
        seg_auc_score = roc_auc_score(masks.ravel(), scores.ravel())
        seg_pr_score = average_precision_score(masks.ravel(), scores.ravel())
        # metrics over all data
        print(f"Det AUC: {det_auc_score:.4f}, Seg AUC: {seg_auc_score:.4f}")
        print(f"Det PR: {det_pr_score:.4f}, Seg PR: {seg_pr_score:.4f}")
        
        # save detection metrics
        with open(os.path.join(self.eval_path, 'det_pr_auc.csv'), mode='w') as f:
                f.write("det_pr, det_auc\n")
                f.write(f"{det_pr_score:.5f},{det_auc_score:.5f}") 
            
