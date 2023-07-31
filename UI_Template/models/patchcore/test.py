import torch
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import datasets, models, transforms
import onnxruntime


#mode_onnx = False

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize([512,512]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_dataset_val = datasets.ImageFolder('./data/mvtec/test' ,data_transforms['val'])
dataloader_val = torch.utils.data.DataLoader(image_dataset_val, batch_size=1, shuffle=False, num_workers=0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_onnx = onnxruntime.InferenceSession("resnet-18-anomaly.onnx")

model = torch.load("resnet-18-anomaly.pth",map_location=device)
model.eval()

for i , (inputs, labels) in enumerate(dataloader_val):    
    name = image_dataset_val.imgs[i][0]
    ort_inputs = {model_onnx.get_inputs()[0].name: to_numpy(inputs)}
    output_onnx = model_onnx.run(None, ort_inputs)[0]
    map_onnx,score_onnx = output_onnx[0:np.shape(output_onnx)[0]//2,:,:,:][0][0],output_onnx[np.shape(output_onnx)[0]-1,0,0,0]

    inputs = inputs.to(device)
    output = model(inputs).cpu().numpy()
    map,score = output[0:np.shape(output)[0]//2,:,:,:][0][0],output[np.shape(output)[0]-1,0,0,0]
    #scores = model(inputs)
    #score = scores[0].cpu().item()
    #print(torch.sum(map)/score)
    if score - 1.377 > 0.2:
        tag = "NG"
    else: 
        tag = "PASS"
    
    
    
    #map = map[0].cpu().numpy()[0]
    map = map / map.max()
    map = map * 255
    map = map.astype(np.uint8)
    map = cv2.cvtColor(map,cv2.COLOR_GRAY2BGR)
    # Difference of two image
    #map_onnx = map_onnx[0].cpu().numpy()[0]
    map_onnx = map_onnx / map_onnx.max()
    map_onnx = map_onnx * 255
    map_onnx = map_onnx.astype(np.uint8)
    map_onnx = cv2.cvtColor(map_onnx,cv2.COLOR_GRAY2BGR)

    map_residule = map - map_onnx

    img = cv2.resize(cv2.imdecode(np.fromfile(name, dtype=np.uint8),-1),(map.shape[0],map.shape[1]))
    output =  np.vstack((img, map))
    cv2.imshow("name",output)
    cv2.waitKey(0)
    print('residule presnetage of two maps: ',np.sum(map_residule)/np.sum(map))
    print('residule presnetage of two scores: ',(score_onnx-score)/score)
    

    print("label:" + str(labels) + " predict: " + tag  + " score:" + str(score))
    