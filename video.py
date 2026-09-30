import time
from datetime import datetime
import cv2
from picamera2 import Picamera2

# 1. 檔名以日期與時間命名
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"{timestamp}.mp4"

# 2. 初始化相機
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={'size': (640, 480)})
picam2.configure(video_config)
picam2.start()

# 3. 設定影片寫入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_filename, fourcc, 30.0, (640, 480))

record_duration = 10.0  # 錄製 10 秒
start_time = time.time()
prev_time = start_time
fps = 0.0

print(f"開始錄影（10 秒後自動停止），檔名：{output_filename}")

while (time.time() - start_time) < record_duration:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 【選用】影像導正（若畫面顛倒請解除註解）
    # frame = cv2.rotate(frame, cv2.ROTATE_180)

    # 計算動態 FPS
    current_time = time.time()
    time_diff = current_time - prev_time
    prev_time = current_time

    if time_diff > 0:
        fps = 1.0 / time_diff

    # --- 左上角：動態 FPS ---
    fps_text = f"FPS={fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),  # 綠色
        2,
        cv2.LINE_AA
    )

    # --- 右上角：日期與時間 ---
    date_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    
    # 計算文字寬度以精準靠右對齊 (寬度為 640)
    (text_w, text_h), _ = cv2.getTextSize(date_text, font, font_scale, font_thickness)
    date_x = 640 - text_w - 20  # 距離右邊緣保留 20px
    date_y = 35

    cv2.putText(
        frame,
        date_text,
        (date_x, date_y),
        font,
        font_scale,
        (255, 255, 255),  # 白色文字
        font_thickness,
        cv2.LINE_AA
    )

    # 寫入影片幀
    video_writer.write(frame)

# 4. 釋放資源
video_writer.release()
picam2.stop()

print(f"錄影完成！影片已儲存至：{output_filename}")