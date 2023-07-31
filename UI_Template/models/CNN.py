import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
from PyQt5.QtCore import QObject, pyqtSignal
import pickle
import numpy as np
import time
import cv2


class CNN_ResNet(QObject):
    # Callback function
    finished = pyqtSignal()
    sendinfo = pyqtSignal(str)
    progress = pyqtSignal(str)
    sendresult = pyqtSignal(str)

    def __init__(self, cfg):
        super(CNN_ResNet, self).__init__()
        self.cfg = cfg
        self.model_path = cfg.save_path    # model and results saving path
        self.n_layers = len(cfg.cnn_layers)
        self.n_dim = cfg.latent_dim
        self.log_step = 10
        self.data_name = cfg.data_name
        self.img_size = cfg.img_size
        self.threshold = cfg.thred
        self.device = torch.device(cfg.device)
        self.forced_stop = False
        self.detect_imgs = []

        # Define the transformation to apply to the images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the input size of ResNet50
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Load the dataset from the folder
        self.batch_size = 32
        self.train_data_path = cfg.train_data_path
        self.test_data_path = cfg.test_data_path
        
        if self.train_data_path and self.train_data_path!="":
            self.train_data = datasets.ImageFolder(root=self.train_data_path, transform=self.transform)
            self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.test_data = datasets.ImageFolder(root=self.test_data_path, transform=self.transform)
            self.test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        # Create the ResNet50 model
        self.model = models.resnet50(pretrained=True)  # Load pre-trained weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        print('Start training (Epoch = {}, lr = {})'.format(self.cfg.epochs, self.cfg.lr))
        self.sendinfo.emit('Start training (Epoch = {}, lr = {})'.format(self.cfg.epochs, self.cfg.lr))

        # Modify the classifier layer for your specific dataset
        num_classes = len(self.train_data.classes)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=float(self.cfg.lr))

        # Train the model
        num_epochs = self.cfg.epochs  # total epochs
        self.model.to(self.device)
        print(self.device)

        iters_per_epoch = len(self.train_data_loader)
        self.progress.emit(str(num_epochs) +',' +str(iters_per_epoch*num_epochs) + ',*')

        step = 0
        for epoch in range(num_epochs):
            
            # Training
            self.model.train()
            running_loss = 0.0
            
            local_step = 0
            for inputs, labels in self.train_data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                step +=1
                local_step +=1
                self.progress.emit(str(epoch+1)+','+str(step)+','+str(running_loss / local_step)+',0')
            #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / local_step}")

            # Validation
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in self.test_data_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f"Validation Accuracy after {epoch + 1} epochs: {accuracy:.2f}")

            self.progress.emit('0,' + str(step)+ ',0,' + str(accuracy))


        # Save the trained model
        label_map = {i: class_name for i, class_name in enumerate(self.train_data.classes)}
        labelmap_path = os.path.join(self.model_path, 'labelmap.pkl')
        with open(labelmap_path, 'wb') as f:
            pickle.dump(label_map, f)
        save_path = os.path.join(self.model_path, 'model.pt')
        torch.save(self.model.state_dict(), save_path)
        self.sendinfo.emit("Done.")

    def load_model(self, path=None):
        print("Loading model...")
        if path is None:
            model_path = os.path.join(self.model_path)
        else:
            model_path = path
        print("model path:", model_path)
        if not os.path.exists(model_path):
            print("Model not exists.")
            return False
        
        labelmap_path = model_path.replace('model.pt','labelmap.pkl')
        with open(labelmap_path, 'rb') as f:
            self.label_map = np.load(f,allow_pickle = True)
        num_classes = len(self.label_map)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded:", model_path)
        return True


    # Use the model to classify an image in the computer
    def classify_image(self, image_path, img = None):
        if image_path:
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_tensor = self.transform(image).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            outputs = outputs.detach().cpu().numpy()[0]
            outputs = np.exp(outputs)/np.exp(outputs).sum()
            score = np.max(outputs)
            predicted_class = np.argmax(outputs)
            class_idx = predicted_class.item()
            class_name = self.label_map[class_idx]
            return class_name,score
        
    def create_verify_list(self, folder):
        self.verify_list = []
        for root, _,files  in os.walk(folder):
            for f in files:
                if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp'):
                    self.verify_list.append(os.path.join(root,f)) 
        
    def verify(self):
        if self.load_model(self.cfg.model_name):
            print("Model Loaded.")
            self.sendinfo.emit("Model Loaded.")
        else:
            print("None pretrained models.")
            return
        self.sendinfo.emit("Start to verify.")
        self.create_verify_list(self.test_data_path)
        self.sendresult.emit("{0}^U^{1}^U^{2}".format('init','',str(len(self.verify_list))))

        for f in self.verify_list:
            class_name,score = self.classify_image(f)
            self.sendresult.emit("{0}^U^{1}^U^{2}".format(f.replace('\\','/'),class_name,str(score)))
            
        
        self.sendinfo.emit("Task done.")
        self.sendresult.emit("{0}^U^".format('Done'))

    
    def detect(self):
        if self.load_model(self.cfg.model_name):
            print("Model Loaded.")
            self.sendinfo.emit("Model Loaded.")
        else:
            print("None pretrained models.")
            return
        self.sendinfo.emit("Start to detect.")
        count = 0
        while True:
            if len(self.detect_imgs)==0:
                self.sendresult.emit("no img")
                time.sleep(0.001)
                continue
            else:
                image = self.detect_imgs.pop(0)
                class_name,score = self.classify_image(image_path=None,img=image)
                self.sendresult.emit("{0}^U^{1}^U^{2}".format('None',class_name,str(score)))
                count+=1