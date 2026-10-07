# Raspberry Pi 電腦視覺初體驗 (First Assignment)

這是我第一次作業的儲存庫，主要使用 Raspberry Pi 搭配 `picamera2` 與 `OpenCV` 來進行各種電腦視覺的基礎應用。

## 專案結構與檔案說明

* **`photo.py`**
  使用 `picamera2` 拍攝一張照片，將其轉換為灰階影像後，儲存為 `gray_capture_0922.jpg`。
* **`video.py`**
  使用 `picamera2` 錄製 10 秒鐘的影片。影片的左上角會即時顯示 FPS，右上角會顯示當下的日期與時間，最後存成 MP4 檔案。
* **`face.py`**
  讀取一張圖片 (`photo.jpg`)，利用 OpenCV 內建的 Haar 特徵分類器 (`haarcascade_frontalface_alt.xml`) 來進行人臉偵測，並在偵測到的人臉周圍畫上綠色方框。
* **`lanedetect.py`**
  結合攝影機即時畫面進行「車道線偵測」。步驟包含：擷取 ROI (感興趣區域)、過濾白線、Canny 邊緣偵測以及 Hough 直線轉換，最後將標記出的車道線與原畫面疊加並錄製成影片。

## 執行環境與套件需求

本專案主要運行於 Raspberry Pi 環境，並需要以下套件：
- Python 3.x
- `opencv-python` (cv2)
- `picamera2`
- `numpy`

可以透過以下指令安裝（若在 Raspberry Pi OS 上，`picamera2` 通常已內建）：
```bash
pip install -r requirements.txt
```

## 執行方式範例

執行各個程式前，請確保攝影機已正確連接並啟用。

```bash
# 拍攝灰階照片
python photo.py

# 錄製帶有時間與 FPS 的影片
python video.py

# 執行人臉偵測 (需準備一張 photo.jpg 在同目錄下)
python face.py

# 執行車道線偵測即時錄影
python lanedetect.py
```