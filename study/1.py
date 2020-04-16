import cv2
import numpy as np


lena = cv2.imread("lena.jpg", -1)  # 读取lena图片： cv2.imread("file_name"[, flags])
cv2.namedWindow("demo")  # 创建窗口名称: cv2.namedWindow("file_name")
cv2.imshow("demo", lena)  # 显示图像： cv2.imshow("win_name", file_name)
img = cv2.imread("lena.jpg", 0)  # 装换为灰度图
cv2.imshow("demo_before", img)
# 修改部分区域像素为白色
for i in range(10, 100):
    for j in range(80, 100):
        img[i, j] = 255
cv2.imshow("demo_after", img)
key = cv2.waitKey()  # 等待按键: key = cv2.waitKey([delay])
if key == ord('A'):
    cv2.imwrite("demo_before.jpg", img)  # 保存图片： cv2.imwrite("file_name",img [, params])
    cv2.imwrite("demo_after.jpg", img)
elif key == ord('B'):
    cv2.destroyAllWindows()  # 销毁窗口： cv2.destorywindow("win_name")



