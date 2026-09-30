import cv2

# 載入預訓練的 Haar 人臉分類器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
)

# 讀取影像並轉為灰階（Haar 偵測通常在灰階影像上執行以提高效能）
img = cv2.imread('photo.jpg')

if img is None:
    print("找不到指定的圖片檔案，請確認路徑是否正確！")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 執行人臉偵測
# scaleFactor=1.1: 每次影像縮小的比例
# minNeighbors=3: 候選矩形保留所需的最小相鄰矩形數量
# minSize=(30, 30): 最小搜尋人臉尺寸（可依需求加入）
faces = face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1, 
    minNeighbors=3, 
    minSize=(30, 30)
)

print(faces)
print(f"------> Found {len(faces)} faces! <------")

# 在每張偵測到的人臉周圍繪製綠色矩形框 (B, G, R)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 顯示標註後的影像視窗
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 若要儲存結果，可解除註解下一行：
# cv2.imwrite('detected_faces.jpg', img)