import os
import cv2

def collect_faces(user_name, sample_count=50):
    output_dir = os.path.join("data", user_name)
    os.makedirs(output_dir, exist_ok=True)

    # 載入 Haar Cascade 分類器
    cascade_path = "haarcascade_frontalface_alt.xml"
    if not os.path.exists(cascade_path):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(0)

    print(f"[INFO] 開始收集 {user_name} 的人臉樣本，請正對攝像頭...")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 無法取得攝影機畫面")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100))

        for (x, y, w, h) in faces:
            count += 1
            # 裁切灰階人臉並正規化大小 (200x200)
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            img_path = os.path.join(output_dir, f"{count:03d}.jpg")
            cv2.imwrite(img_path, face_roi)

            # 繪製辨識框與採集進度
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Collected: {count}/{sample_count}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Collect Faces - Press 'q' to Quit", frame)

        if count >= sample_count or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 收集完成！共儲存 {count} 張影像於 {output_dir}")

if __name__ == "__main__":
    name = input("請輸入人員 ID/名稱 (例如 student01): ").strip()
    if name:
        collect_faces(name)