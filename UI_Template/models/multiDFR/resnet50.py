import torch
import torch.nn as nn
from torchvision.models.resnet import resnext101_32x8d
from torchvision.models.mobilenetv3 import mobilenet_v3_large

class RESNET50W(torch.nn.Module):
    def __init__(self, gradient=False):
        super(RESNET50W, self).__init__()
        def hook_t(module, input, output):
            self.features.append(output)
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)

        for param in self.model.parameters():
                param.requires_grad = False
  
        self.model.layer1[-1].register_forward_hook(hook_t)
        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t) 
        self.model.layer4[-1].register_forward_hook(hook_t) 
          

    def init_features(self):
        self.features = []

    def forward(self, x_t,feature_layers):
        self.init_features()
        _ = self.model(x_t)
        out = {'relu1': self.features[0], 'relu2': self.features[1],'relu3': self.features[2],'relu4': self.features[3]}
        return dict((key, value) for key, value in out.items() if key in feature_layers)

class RESNET50(torch.nn.Module):
    def __init__(self, gradient=False):
        super(RESNET50, self).__init__()
        def hook_t(module, input, output):
            self.features.append(output)
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)

        for param in self.model.parameters():
                param.requires_grad = False
  
        self.model.layer1[-1].register_forward_hook(hook_t)
        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t) 
        self.model.layer4[-1].register_forward_hook(hook_t) 
          

    def init_features(self):
        self.features = []

    def forward(self, x_t,feature_layers):
        self.init_features()
        _ = self.model(x_t)
        out = {'relu1': self.features[0], 'relu2': self.features[1],'relu3': self.features[2],'relu4': self.features[3]}
        return dict((key, value) for key, value in out.items() if key in feature_layers)

class RESNEXT50(torch.nn.Module):
    def __init__(self, gradient=False):
        super(RESNEXT50, self).__init__()
        def hook_t(module, input, output):
            self.features.append(output)
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnext50_32x4d', pretrained=True)

        for param in self.model.parameters():
                param.requires_grad = False
  
        self.model.layer1[-1].register_forward_hook(hook_t)
        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t) 
        self.model.layer4[-1].register_forward_hook(hook_t) 
          

    def init_features(self):
        self.features = []

    def forward(self, x_t,feature_layers):
        self.init_features()
        _ = self.model(x_t)
        out = {'relu1': self.features[0], 'relu2': self.features[1],'relu3': self.features[2],'relu4': self.features[3]}
        return dict((key, value) for key, value in out.items() if key in feature_layers)

class RESNEXT101(torch.nn.Module):
    def __init__(self, gradient=False):
        super(RESNEXT101, self).__init__()
        def hook_t(module, input, output):
            self.features.append(output)
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnext101_32x8d', pretrained=True)

        for param in self.model.parameters():
                param.requires_grad = False
  
        self.model.layer1[-1].register_forward_hook(hook_t)
        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t) 
        self.model.layer4[-1].register_forward_hook(hook_t) 
          

    def init_features(self):
        self.features = []

    def forward(self, x_t,feature_layers):
        self.init_features()
        _ = self.model(x_t)
        out = {'relu1': self.features[0], 'relu2': self.features[1],'relu3': self.features[2],'relu4': self.features[3]}
        return dict((key, value) for key, value in out.items() if key in feature_layers)

class MOBILENETV3L(torch.nn.Module):
    def __init__(self, gradient=False):
        super(MOBILENETV3L, self).__init__()
        def hook_t(module, input, output):
            self.features.append(output)
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v3_large', pretrained=True)

        for param in self.model.parameters():
                param.requires_grad = False
  
        self.model.features[3].register_forward_hook(hook_t)
        self.model.features[6].register_forward_hook(hook_t)
        self.model.features[12].register_forward_hook(hook_t) 
          

    def init_features(self):
        self.features = []

    def forward(self, x_t,feature_layers):
        self.init_features()
        _ = self.model(x_t)
        out = {'relu1': self.features[0], 'relu2': self.features[1],'relu3': self.features[2]}
        return dict((key, value) for key, value in out.items() if key in feature_layers)

class MOBILENETV3S(torch.nn.Module):
    def __init__(self, gradient=False):
        super(MOBILENETV3S, self).__init__()
        def hook_t(module, input, output):
            self.features.append(output)
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v3_small', pretrained=True)

        for param in self.model.parameters():
                param.requires_grad = False
  
        self.model.features[3].register_forward_hook(hook_t)
        self.model.features[6].register_forward_hook(hook_t)
        self.model.features[12].register_forward_hook(hook_t) 
          

    def init_features(self):
        self.features = []

    def forward(self, x_t,feature_layers):
        self.init_features()
        _ = self.model(x_t)
        out = {'relu1': self.features[0], 'relu2': self.features[1],'relu3': self.features[2]}
        return dict((key, value) for key, value in out.items() if key in feature_layers)