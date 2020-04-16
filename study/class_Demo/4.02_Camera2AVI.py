import cv2 as cv
import numpy as np


def main():
    video_demo()


def video_demo():
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        print("Camera is not ready!")
        capture.release()
        return
    fourcc = cv.VideoWriter_fourcc(*'XVID')#指定视频格式
    out = cv.VideoWriter('d:\output.avi', fourcc, 20, (640, 480))
    while(True):
        ret, frame = capture.read()
        # get_image_info(frame)
        if ret == True:
            frame = cv.flip(frame, 1)
            cv.imshow("video", frame)
            out.write(frame)
            c = cv.waitKey(50)
            if c == 27:     #esc退出
                break
        else:
            break


def get_image_info(image):
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)
    pixel_data = np.array(image)
    print(pixel_data)


if __name__ == '__main__':
    main()

