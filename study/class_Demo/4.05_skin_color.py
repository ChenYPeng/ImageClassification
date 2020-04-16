import cv2 as cv
import numpy as np


def skin_color():
    capture = cv.VideoCapture("pic/output.avi")
    while True:
        ret, frame = capture.read()
        if not ret:
            break;
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 30, 80])  # 肤色范围
        upper_hsv = np.array([17, 170, 255])
        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        dst = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow("video", frame)
        cv.imshow("mask", dst)
        c = cv.waitKey(40)
        if c == 27:
            break


skin_color()

