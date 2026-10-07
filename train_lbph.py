import os
import json
import cv2
import numpy as np

def train():
    data_dir = "data"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    faces = []
    labels = []
    label_map = {}
    current_id = 0

    print("[INFO] 正在讀取資料集並提取特徵...")
    for entry in sorted(os.listdir(data_dir)):
        person_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(person_dir):
            continue

        label_map[current_id] = entry
        for img_name in os.listdir(person_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(cv2.resize(img, (200, 200)))
                    labels.append(current_id)

        current_id += 1

    if not faces:
        print("[ERROR] 未找到任何訓練人臉影像，請先執行 collect_faces.py")
        return

    # 建立 LBPH 人臉辨識器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    # 儲存模型與標籤對應表
    model_path = os.path.join(models_dir, "lbph_model.yml")
    labels_path = os.path.join(models_dir, "labels.json")

    recognizer.write(model_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] 訓練完成！模型已儲存至 {model_path}，標籤表儲存至 {labels_path}")

if __name__ == "__main__":
    train()