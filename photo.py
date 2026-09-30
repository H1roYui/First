from picamera2 import Picamera2
import cv2
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(
main={'size': (640,480)}))
picam2.start()
frame = picam2.capture_array()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_capture_0922.jpg', gray)
print(' 影像已存檔 gray_capture_0922.jpg')