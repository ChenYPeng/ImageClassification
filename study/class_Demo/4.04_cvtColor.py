import cv2 as cv
import numpy as np


def main():
    src = cv.imread("pic/lena.png")
    color_space_demo(src)
    cv.waitKey(0)
    cv.destroyAllWindows()


def color_space_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)
    Ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    cv.imshow("ycrcb", Ycrcb)

    b, g, r = cv.split(image)   # 通道分离
    cv.imshow("blue", b)
    cv.imshow("green", g)
    cv.imshow("red", r)
    src = cv.merge([b, g, r])
    src[:, :, 0] = 0    # 去掉红色
    cv.imshow("changed image", src)


if __name__ == '__main__':
    main()