# CNN Tool Template

本專案將展示如何製作簡易的 CNN 模型訓練工具，使用者亦可直接將本專案下載後修改。

本說明將分為四個部分: 包含:
1. Anaconda 下載與安裝
2. 開發環境建置
3. 範例程式下載與應用
4. 程式修改指南

## 1. Anaconda 下載與安裝
為了建置獨立的開發環境，建議使用者安裝 Anaconda。請至 Anaconda 官網下載 https://www.anaconda.com/download 並完成安裝。
![image](/img/1.jpg)


## 2. 開發環境建置
開啟 "Anaconda Prompt" ，使用以下指令創建環境:
```
conda create --name cnn_template python=3.7
```

輸入以下指令，即可使用使用剛才創建的環境:
```
conda activate cnn_template
```

輸入以下指令即可開啟 Visual Studio Code:

註: 如果沒有安裝 Visual Studio Code 請至官網下載: https://code.visualstudio.com/
```
code
```
![image](/img/5.jpg)

## 3.範例程式下載與應用
下載本範例程式，以 Visual Studio Code 開啟

由 menu.py 進入，即可開始使用本範例程式
本程式提供 training tool, verifying tool, detecting tool 三種工具可使用

### Training tool
Training tool 的功能是使用指定的資料集訓練 CNN 模型，並儲存至指定路徑。
開啟 training tool 後，點擊左方 Run (三角形) 按鍵，即可開始訓練。使用者亦可針對需求更改上方參數與路徑。
![image](/img/train.jpg)

### Verifying tool
Verifying tool 的功能是使用指定的資料集驗證訓練後的 CNN 模型。
開啟 verifying tool 後，點擊左方 Run (三角形) 按鍵，即可開始驗證。使用者亦可針對需求更改上方參數與路徑。
![image](/img/verify.jpg)

### Detecting tool
Detecting tool 的功能是使用指定的 CNN 模型檢測網路攝影機 (Web Cam) 的輸入影像。
開啟 detecting tool 後，點擊左方第一個 Run (三角形) 按鍵，即可連結攝影機，再點擊第二個 Run 按鍵，即可開始 AI 檢測功能。使用者亦可針對需求更改模型路徑。
![image](/img/detect.jpg)

註: 若出現找不到套件的情形，請自行安裝缺少的套件包，參考版本如下 (PyTorch 可以使用 Conda install，其餘使用pip install 即可)

```
altgraph==0.17.3
certifi==2022.12.7
click==8.1.3
colorama==0.4.6
coloredlogs==15.0.1
cycler==0.11.0
faiss-cpu==1.7.4
filelock==3.11.0
Flask==2.2.5
flatbuffers==23.3.3
fonttools==4.38.0
huggingface-hub==0.13.4
humanfriendly==10.0
imageio==2.28.1
importlib-metadata==6.3.0
itsdangerous==2.1.2
Jinja2==3.1.2
joblib==1.2.0
kiwisolver==1.4.4
kornia==0.6.9
MarkupSafe==2.1.2
matplotlib==3.5.3
mpmath==1.3.0
networkx==2.6.3
numpy @ file:///D:/bld/numpy_1649806536290/work
onnx==1.13.1
onnxconverter-common==1.13.0
onnxruntime==1.14.1
onnxruntime-gpu==1.14.1
opencv-python==4.7.0.72
packaging==23.0
pandas==1.1.5
pefile==2023.2.7
Pillow @ file:///D:/bld/pillow_1660386017560/work
protobuf==3.20.3
pyinstaller==5.11.0
pyinstaller-hooks-contrib==2023.3
pyminizip==0.2.6
pyparsing==3.0.9
PyQt5==5.15.9
PyQt5-Qt5==5.15.2
PyQt5-sip==12.12.1
PyQtChart==5.15.6
PyQtChart-Qt5==5.15.2
pyreadline==2.1
PySide2==5.15.2.1
python-dateutil==2.8.2
pytz==2023.3
PyWavelets==1.3.0
pywin32-ctypes==0.2.0
PyYAML==6.0
scikit-image==0.19.3
scikit-learn==1.0.2
scipy==1.7.3
shiboken2==5.15.2.1
six==1.16.0
sklearn==0.0.post1
sympy==1.10.1
threadpoolctl==3.1.0
tifffile==2021.11.2
timm==0.6.13
torch==1.12.1
torchaudio==0.12.1
torchvision==0.13.1
tqdm==4.65.0
Werkzeug==2.2.3
wincertstore==0.2
zipp==3.15.0
```

## 4. 程式修改指南

### 程式碼修改
若欲進行程式碼修改，可至專案資料夾中開啟對應的 .py 檔案，其對應如下:
```
Menu > menu.py
Training tool > train.py
Verifying tool > verify.py
Detecting tool > detect.py
```

此類檔案一般分為三個區塊:
1. 建構子 : 初始化 UI 介面與類別中之成員
2. 元件事件 : 將 UI 元件的事件與函式連結
3. 函式 : 與元件連結的函式功能

只要針對想修改的功能，分別修改三個區塊即可。

![image](/img/example.jpg)

### Qt Designer UI 修改
若欲進行 UI 修改，請至 https://build-system.fman.io/qt-designer-download 下載 Qt Designer 並完成安裝。
![image](/img/2.jpg)

使用 QT Designer 可以打開範例程式中的.ui檔案，使用者可以依照需求對內容設計進行更改。
QT Designer 的說明與使用方式可以參考以下教學: https://www.youtube.com/watch?v=_E4Rj4I58m0&list=PL0uF99K6uCYPp2hv99pcUPpkUtiDWjGdW&index=4

註 1: 如果不想使用  QT Designer 內建的背景圖式以及元件底色，可以從外部匯入圖片再進行使用，建議以PPT繪製再輸出為png檔案，再按照以下方法使用圖片:
1. 將輸出的 png 檔案移到 ui_img 資料夾中
2. 開啟 pic.qrc 檔案並將新增的圖片路徑加到檔案內
![image](/img/3.jpg)
4. 儲存 pic.qrc 檔案並關閉
5. 在 QT Designer 中對元件按下右鍵，開啟 styleSheet 並將圖片路徑更改為新的路徑即可
![image](/img/4.jpg)

註2: 若有修改 pic.qrc 檔案，請在工作目錄下開啟 CMD 並執行以下程式碼:

```
pyrcc5 -o pic.py pic.qrc
```
註3: 若有修改 train.ui 檔案，請在工作目錄下開啟 CMD 並執行以下程式碼: (其他ui檔案修改依此類推)
```
pyuic5 train.ui -o train_ui.py
```
輸出完 train_ui.py 之後，記得將該檔案最後一列之 pic_rc 修改為 pic


