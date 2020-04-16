import cv2 as cv
import numpy as np


def main():
    avi_demo()


def avi_demo():
    capture = cv.VideoCapture("pic/output.avi")
    if not capture.isOpened():
        print("File is not exit!")
        capture.release()
        return
    while(True):
        ret, frame = capture.read()
        if ret == True:
            frame = cv.flip(frame, 1)
            cv.imshow("video", frame)
            c = cv.waitKey(50)
            if c == 27:     #esc退出
                break
        else:
            break



if __name__ == '__main__':
    main()

