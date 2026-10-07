import os
import json
import cv2

def run_recognition(threshold=75):
    model_path = os.path.join("models", "lbph_model.yml")
    labels_path = os.path.join("models", "labels.json")

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        print("[ERROR] 找不到模型檔或標籤表，請先執行 train_lbph.py")
        return

    # 載入 LBPH 模型與標籤映射
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    with open(labels_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    # 將 key 轉回整數型態
    label_map = {int(k): v for k, v in label_map.items()}

    # 載入 Haar 分類器
    cascade_path = "haarcascade_frontalface_alt.xml"
    if not os.path.exists(cascade_path):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    print("[INFO] 即時辨識啟動中，按 'q' 鍵退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label_id, distance = recognizer.predict(face_roi)

            # LBPH 距離越小相似度越高；小於閾值代表匹配成功
            if distance <= threshold and label_id in label_map:
                name = label_map[label_id]
                color = (0, 255, 0)
                text = f"{name} ({distance:.1f})"
            else:
                name = "Unknown"
                color = (0, 0, 255)
                text = f"{name} ({distance:.1f})"

            # 繪製結果
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("LBPH Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition(threshold=75)