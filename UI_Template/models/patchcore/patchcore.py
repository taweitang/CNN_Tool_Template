import torch
import numpy as np
from tqdm import tqdm
from models.patchcore.patch_core import PatchcoreModel
from torchvision import datasets, models, transforms
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import onnxruntime
import shutil
import os
import cv2
import json
from zipfile import ZipFile
import time


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class patchcore(QObject):

    # Callback function
    finished = pyqtSignal()
    sendinfo = pyqtSignal(str)
    progress = pyqtSignal(str)
    sendresult = pyqtSignal(str)

    def __init__(self, cfg):
        super(patchcore, self).__init__()

        print('init....')

        self.cfg = cfg
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([self.cfg['size'],self.cfg['size']]),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor()
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize([self.cfg['size'],self.cfg['size']]),
                transforms.ToTensor()
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        
        self.istrain = cfg['train']
        self.test_path = cfg['test_data_path']
        if self.test_path != "":
            self.image_dataset_val = datasets.ImageFolder(self.test_path ,self.data_transforms['val'])
            self.dataloader_val = torch.utils.data.DataLoader(self.image_dataset_val, batch_size=1, shuffle=False, num_workers=0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detect_imgs = []
        self.sub_index = 0

        if self.istrain:
            backbone_list = ['resnet18','resnet50','wide_resnet50_2']
            self.train_path = cfg['train_data_path'][:-5]
            self.k = cfg['k']
            self.ratio = cfg['ratio']
            self.image_dataset = datasets.ImageFolder(self.train_path ,self.data_transforms['train'])
            self.dataloader = torch.utils.data.DataLoader(self.image_dataset, batch_size=1, shuffle=True, num_workers=0)
            self.model = PatchcoreModel((self.cfg['size'],self.cfg['size']),["layer2","layer3"],backbone_list[self.cfg['backbone']],callback = self.callback_function)
            self.model = self.model.to(self.device)
            self.embeddings = []
            self.result = np.array([])
            self.mean ,self.std = 0,0

    def callback_function(self, a,b):
        self.sub_index +=1
        self.progress.emit(str(self.sub_index)+',' + str(b) + ',2')

    def create_embeddings(self):

        print('create embeddings...')
        self.progress.emit(str(len(self.dataloader)) +',' +str(len(self.dataloader)) + ',' + str(len(self.dataloader)) + ',*')
        i = 1
        for inputs, labels in tqdm(self.dataloader):
            inputs = inputs*255 
            inputs = inputs.to(self.device)
            embedding = self.model(inputs)
            self.embeddings.append(embedding)
            self.progress.emit(str(i)+',1')
            i+=1

        self.embeddings = torch.vstack(self.embeddings)
        self.sub_index = 0
        self.model.subsample_embedding(self.embeddings, self.ratio, self.k)
        self.model.eval()


    def first_testing(self):

        print('start to test')
        i = 1
        for inputs, labels in tqdm(self.dataloader):
            inputs = inputs*255 
            inputs = inputs.to(self.device)

            #output = self.model(inputs).cpu().numpy()
            #scores = output[np.shape(output)[0]//2:np.shape(output)[0],0,0,0]

            anomaly_map,scores = self.model(inputs)


            self.result = np.append(self.result, scores)
            self.progress.emit(str(i)+',3')
            i+=1

            #self.sendinfo.emit('Epoch {}/{}'.format(epoch, epochs) + ", {}: {:.4f}".format(tag, value))

        self.mean = np.mean(self.result)
        self.std = np.std(self.result)
        self.max = self.result.max()
        self.min = self.result.min()
        print("result:")
        print(self.result)
        print ("mean:" + str(self.mean.item()))
        print ("std:" + str(self.std.item()))

    def first_validation(self):

        print('start to validate')
        for i, (inputs, labels) in enumerate(tqdm(self.dataloader_val)):

            name = self.image_dataset_val.imgs[i][0]
            inputs = inputs*255 
            inputs = inputs.to(self.device)

            #map, scores = model(inputs)
            output = self.model(inputs).cpu().numpy()
            map,score = output[0:np.shape(output)[0]//2,:,:,:],output[0,0,0,0]
            #score = scores[0].cpu().item()
            if score - self.mean > 3 * self.std:
                tag = "NG"
            else:
                tag = "PASS"

            #print("label:" + str(labels) + " predict: " + tag  + " score:" + str(score) + " file:"+ name)

    def verify(self):
        self.sendresult.emit("{0}^U^{1}^U^{2}".format('init','',str(len(self.dataloader_val))))
        self.model_onnx = onnxruntime.InferenceSession(self.cfg['model_path'])
        shutil.rmtree('./heatmapTemp/')
        os.mkdir('./heatmapTemp/')
        max_score = 0
        for i, (inputs, labels) in enumerate(tqdm(self.dataloader_val)):
            name = self.image_dataset_val.imgs[i][0]
            ort_inputs = {self.model_onnx.get_inputs()[0].name: (to_numpy(inputs)*255).astype('uint8')}
            #output_onnx = self.model_onnx.run(None, ort_inputs)[0]
            #map_onnx,score_onnx = output_onnx[0:np.shape(output_onnx)[0]//2,:,:,:][0][0],output_onnx[np.shape(output_onnx)[0]-1,0,0,0]
            output_onnx = self.model_onnx.run(None, ort_inputs)
            map_onnx,score_onnx = output_onnx[0][0][0],output_onnx[1][0]
            map_onnx = map_onnx / map_onnx.max()
            map_onnx = map_onnx * 255
            map_onnx = map_onnx.astype(np.uint8)
            map_onnx = cv2.applyColorMap(np.uint8(map_onnx), cv2.COLORMAP_JET)
            #map_onnx = cv2.cvtColor(map_onnx,cv2.COLOR_GRAY2BGR)

            heatmap_path = name.replace(".png","_ht.png").replace(".jpg","_ht.png")
            heatmap_path = heatmap_path.split('\\')[-2] + heatmap_path.split('\\')[-1]
            heatmap_path = './heatmapTemp/' + heatmap_path
            cv2.imwrite(heatmap_path,map_onnx)

            if score_onnx>max_score:
                max_score = score_onnx

            self.sendresult.emit("{0}^U^{1}^U^{2}".format(name.replace('\\','/'),heatmap_path.replace('\\','/'),str(score_onnx)))
        self.sendresult.emit("{0}^U^{1}^U^{2}".format('max','',str(max_score)))
        self.sendinfo.emit("Task done.")

    def heatmap_on_image(self, heatmap, image):
        if heatmap.shape != image.shape:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        out = np.float32(heatmap)/255 + np.float32(image)/255
        out = out / np.max(out)
        return np.uint8(255 * out)

    def detect(self):

        if not os.path.exists(self.cfg['model_path']):
            print("Model not exists.")
            self.sendinfo.emit("Model not exists.")
            return
    
        self.model_onnx = onnxruntime.InferenceSession(self.cfg['model_path'])
        shutil.rmtree('./heatmapTemp/')
        os.mkdir('./heatmapTemp/')

        self.sendinfo.emit("Start to detect.")
        count = 0
        transform = transforms.ToTensor()
        while True:
            if len(self.detect_imgs)==0:
                self.sendresult.emit("no img")
                time.sleep(0.001)
                continue
            else:
                image = self.detect_imgs.pop(0)
                ori_size = image.shape
                if len(ori_size)==2:
                    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
                image_tensor = transform(image)
                image_tensor = transforms.Resize([self.cfg['size'],self.cfg['size']])(image_tensor)
                #image = image.astype(np.float32)/255
                ort_inputs = {self.model_onnx.get_inputs()[0].name: (to_numpy(image_tensor[None, :])*255).astype('uint8')}
                #output_onnx = self.model_onnx.run(None, ort_inputs)[0]
                #map_onnx,score_onnx = output_onnx[0:np.shape(output_onnx)[0]//2,:,:,:][0][0],output_onnx[np.shape(output_onnx)[0]-1,0,0,0]
                output_onnx = self.model_onnx.run(None, ort_inputs)
                map_onnx,score_onnx = output_onnx[0][0][0],output_onnx[1][0]
                map_onnx = map_onnx / map_onnx.max()
                map_onnx = map_onnx * 255
                map_onnx = map_onnx.astype(np.uint8)
                map_onnx = cv2.applyColorMap(np.uint8(map_onnx), cv2.COLORMAP_JET)
                map_onnx = cv2.resize(self.heatmap_on_image(map_onnx,image),[ori_size[1],ori_size[0]])
                heatmap_path = str(count) + "_ht.png"
                #heatmap_path = heatmap_path.split('\\')[-2] + heatmap_path.split('\\')[-1]
                heatmap_path = './heatmapTemp/' + heatmap_path
                #print(heatmap_path)
                cv2.imwrite(heatmap_path,map_onnx)
                self.sendresult.emit("{0}^U^{1}^U^{2}".format('None',heatmap_path.replace('\\','/'),str(score_onnx)))
                count+=1
    
    def save_model(self):
            
        #torch.save(self.model, 'resnet-18-anomaly.pth')
        #print('save: resnet-18-anomaly.pth')


        self.train_meta_data = {}
        self.train_meta_data['N'] = self.k
        self.train_meta_data['size'] = self.cfg['size']
        self.train_meta_data['avg'] = self.mean
        self.train_meta_data['max'] = self.max
        self.train_meta_data['min'] = self.min
        self.train_meta_data['std'] = self.std

        base_name = '3DFADfast'
        json_path = os.path.join(self.cfg['save_path'], base_name +'.json')
        with open(json_path, "w", encoding='utf-8') as outfile:
            json.dump(self.train_meta_data, outfile)

        model_path = os.path.join(self.cfg['save_path'], base_name +'.onnx')
        json_path = os.path.join(self.cfg['save_path'], base_name +'.json')

        #save .onnx
        torch.onnx.export(
            self.model,
            torch.zeros((1, 3, *[self.cfg['size'],self.cfg['size']]),dtype=torch.uint8).to(self.device),
            model_path,
            opset_version=11,
            input_names=["input"],
            output_names=["output_map","output_score"],
        )
        
        with ZipFile(os.path.join(self.cfg['save_path'], 'AD_model.pbz'),'w') as zip:
            # writing each file one by one
            zip.write(model_path,'3DFADfast.onnx')
            zip.write(json_path,'3DFADfast.json')

        os.remove(model_path)
        os.remove(json_path)
        self.sendinfo.emit('Model saved')

    def train(self):
        self.create_embeddings()
        self.first_testing()
        self.save_model()

#model = patchcore(None)
#model.train()