import numpy as np
import cv2

cap = cv2.VideoCapture('http://45.13.210.86:8083/mjpg/video.mjpg')
#cap = cv2.VideoCapture('../../input/video_1.mp4')


while True:
    ret, frame = cap.read()
    cv2.imshow('Salida',frame)
    k = cv2.waitKey(10)&0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()