import cv2
import numpy as np
import sys,os,glob,numpy
from skimage import io

#指定图片的人脸识别然后存储
img = cv2.imread("face2.JPG")
color = (0, 255, 0)
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
classfier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #级联分类器

faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(faceRects) > 0: # 大于0则检测到人脸
    for faceRect in faceRects: # 单独框出每一张人脸
        x, y, w, h = faceRect
        cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3) #5控制绿色框的粗细

# 写入图像
cv2.imwrite('output.jpg',img)
cv2.imshow("Find Faces!",img)
cv2.waitKey(0)