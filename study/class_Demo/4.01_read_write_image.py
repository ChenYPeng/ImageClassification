import cv2 as cv
print("库准备好了")
src = cv.imread("lena.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
cv.imwrite("out.jpg", src)
key = cv.waitKey(0)
if key == ord('a'):
    print('a')
    print(key)
cv.destroyAllWindows()

