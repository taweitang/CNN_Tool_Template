# CNN Tool Template

本專案將展示如何製作簡易的 CNN 模型訓練工具，使用者亦可直接將本專案下載後修改。

本說明將分為四個部分: 包含:
1. Anaconda 下載與安裝
2. Qt Designer UI 設計
3. 開發環境建置
4. 範例程式下載與應用



## 1. Anaconda 下載與安裝
為了建置獨立的開發環境，建議使用者安裝 Anaconda。請至 Anaconda 官網下載 https://www.anaconda.com/download 並完成安裝。
![image](/img/1.jpg)

## 2. Qt Designer UI 設計
安裝好 Anaconda 之後，為了方便進行 UI 設計，請至 https://build-system.fman.io/qt-designer-download 下載 Qt Designer 並完成安裝。
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

## 3. 開發環境建置
設計好 UI 之後，開啟 "Anaconda Prompt" ，使用以下指令創建環境:
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

## 4.範例程式下載與應用
下載本範例程式，以 Visual Studio Code 開啟

由 menu.py 進入，即可開始使用本範例程式

註1: 範例程式中會有註解，如需修改功能只要照註解提示修改即可

註2: 若出現找不到套件的情形，請自行安裝缺少的套件包 (PyTorch 可以使用 Conda install，其餘使用pip install 即可)

註3: 若有修改 pic.qrc 檔案，請在工作目錄下開啟 CMD 並執行以下程式碼:

```
pyrcc5 -o pic.py pic.qrc
```


