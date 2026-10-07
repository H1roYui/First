# LBPH Face Recognition with OpenCV & Haar Cascade

基於 OpenCV 的 Haar 特徵人臉偵測與 LBPH（Local Binary Patterns Histograms）特徵分析之即時人臉辨識系統。

## 📌 專案架構 (Project Structure)

```text
LBPH_Face/
├── collect_faces.py            # 1. 影像收集：透過 Webcam 即時偵測並裁切人臉樣本
├── train_lbph.py               # 2. 模型訓練：讀取人臉集並生成 LBPH 模型與標籤對應檔
├── recognize_lbph.py           # 3. 即時辨識：結合 Webcam 進行 LBPH 特徵比對與標示
├── haarcascade_frontalface_alt.xml  # Haar Cascade 人臉偵測分類器模型檔
├── data/                       # 人臉圖像資料庫
│   ├── student01/              # 學生/人員 01 的灰階人臉照片
│   └── student02/              # 學生/人員 02 的灰階人臉照片
└── models/                     # 訓練結果儲存目錄
    ├── lbph_model.yml          # LBPH 權重與特徵模型檔
    └── labels.json             # 類別標籤與對應姓名檔