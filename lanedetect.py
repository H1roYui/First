import cv2
import numpy as np
import time
from picamera2 import Picamera2
from datetime import datetime

# 1. 建立感興趣區域 (Region of Interest, ROI) 遮罩
def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    
    # 定義只關注車道下半部的梯形區域
    polygons = np.array([[
        (int(width * 0.1), height),             # 左下
        (int(width * 0.4), int(height * 0.6)),  # 左上
        (int(width * 0.6), int(height * 0.6)),  # 右上
        (int(width * 0.9), height)              # 右下
    ]], np.int32)
    
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# 2. 過濾白線：轉換為 HLS 色彩空間擷取高亮度白色區域
def filter_white_lane(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    # 白色的特性：低飽和度、高亮度 (可依光線彈性微調 190)
    lower_white = np.array([0, 190, 0], dtype=np.uint8)
    upper_white = np.array([180, 255, 255], dtype=np.uint8)
    
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    return white_mask

# 3. 將霍夫變換抓到的零散線段繪製成線條
def draw_lane_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is None:
        return line_image
    
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        
        # 依斜率區分左右車道
        if slope < -0.5:
            left_lines.append(line[0])
        elif slope > 0.5:
            right_lines.append(line[0])
            
    # 繪製綠色線條標記
    for line in left_lines + right_lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
    return line_image

# --- 初始化相機 ---
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={'size': (640, 480)})
picam2.configure(video_config)
picam2.start()

# --- 設定錄影為 MP4 格式，檔名加上時間戳記 ---
file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f'lane_detection_{file_timestamp}.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480))

print(f"開始偵測車道線並錄影（按 Ctrl+C 結束）...")
print(f"影片將儲存為: {output_filename}")

# 用於計算即時 FPS
prev_time = time.time()
fps = 0.0

try:
    while True:
        # 計算 FPS
        curr_time = time.time()
        time_diff = curr_time - prev_time
        if time_diff > 0:
            fps = 1.0 / time_diff
        prev_time = curr_time

        # 1. 抓取畫面並轉換成 BGR
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 2. 將畫面旋轉 180 度
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # 3. 取得白線遮罩，並做高斯模糊降噪
        white_mask = filter_white_lane(frame)
        blur = cv2.GaussianBlur(white_mask, (5, 5), 0)

        # 4. Canny 邊緣偵測
        edges = cv2.Canny(blur, 50, 150)

        # 5. 切割出梯形 ROI
        roi_edges = region_of_interest(edges)

        # 6. 霍夫直線轉換
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=40,
            maxLineGap=20
        )

        # 7. 疊加車道線繪製結果
        lane_overlay = draw_lane_lines(frame, lines)
        result = cv2.addWeighted(frame, 0.8, lane_overlay, 1.0, 0)

        # 8. 繪製左上角 FPS（黃色字體）
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(result, fps_text, (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # 9. 繪製右上角 日期與時間（白色字體）
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 動態計算文字寬度，確保文字緊靠右側
        (text_w, _), _ = cv2.getTextSize(current_time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top_right_x = 640 - text_w - 15  # 寬度 640 減去字寬與邊距
        cv2.putText(result, current_time_str, (top_right_x, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # 寫入影片檔案
        video_writer.write(result)

        # 即時預覽（選用）
        # cv2.imshow("Lane Detection", result)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

except KeyboardInterrupt:
    print("\n停止偵測。")

finally:
    video_writer.release()
    print(f"影片已安全儲存為 {output_filename}")
    picam2.stop()
    cv2.destroyAllWindows()