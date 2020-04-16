# 导入OpenCV库
import cv2


# 使用OpenCV2库提供的正面haar级联初始化面级联。这对于图像中的人脸检测是必需的。
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 所需的输出宽度和高度，可根据需要修改。
OUTPUT_SIZE_WIDTH = 700
OUTPUT_SIZE_HEIGHT = 600

# 打开第一个网络摄像机设备
capture = cv2.VideoCapture(0)

# 创建两个名为windows窗口的opencv，用于显示输入、输出图像。
cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

# 把windows窗口靠在一起
cv2.moveWindow("base-image", 20, 200)
cv2.moveWindow("result-image", 640, 200)

# 启动我们正在使用的两个windows窗口的windows窗口线程
cv2.startWindowThread()

# 我们在脸周围画的矩形的颜色
rectangleColor = (0, 100, 255)

while (1):
    # 从网络摄像头中检索最新图像
    rc, fullSizeBaseImage = capture.read()
    # 将图像大小调整为520x420
    baseImage = cv2.resize(fullSizeBaseImage, (520, 420))

    # 检查是否按下了某个键，是否为Q或q，然后销毁所有opencv窗口并退出应用程序，停止无限循环。
    pressedKey = cv2.waitKey(2)
    if (pressedKey == ord('Q')) | (pressedKey == ord('q')):
        cv2.destroyAllWindows()
        exit(0)
    # 结果图像是我们将向用户显示的图像，它是从网络摄像头捕获的原始图像与检测最大人脸的覆盖矩形的组合
    resultImage = baseImage.copy()

    # 我们将使用灰色图像进行人脸检测。
    # 所以我们需要把摄像头捕捉到的基本图像转换成基于灰度的图像
    gray_image = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray_image, 1.3, 5)
    # 由于我们只对“最大”面感兴趣，因此需要计算所找到矩形的最大面积。
    # 为此，首先将所需的变量初始化为0。
    maxArea = 0
    x = 0
    y = 0
    w = 0
    h = 0

    # 在图像中找到的所有面上循环，并检查此面的区域是否是迄今为止最大的
    for (_x, _y, _w, _h) in faces:
        if _w * _h > maxArea:
            x = _x
            y = _y
            w = _w
            h = _h
            maxArea = w * h
    # 如果找到任何面，请在图片中最大的面周围画一个矩形
    if maxArea > 0:
        cv2.rectangle(resultImage, (x - 10, y - 20), (x + w + 10, y + h + 20), rectangleColor, 2)

    # 因为我们想在屏幕上显示比原来的520x420更大的东西，所以我们再次调整图像的大小
    # 请注意，也可以保留基本图像的大版本，并使结果图像成为此大基本图像的副本，并使用缩放因子在右坐标处绘制矩形。
    largeResult = cv2.resize(resultImage, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))
    # 最后，我们在屏幕上显示图像
    cv2.imshow("base-image", baseImage)
    cv2.imshow("result-image", largeResult)