
import cv2
import dlib

# 使用OpenCV库提供的正面haar级联初始化面级联
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 设备输出宽度和高度
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600


def detectAndTrackLargestFace():
    # 打开第一个网络摄像机设备
    capture = cv2.VideoCapture(0)

    # 创建两个名为windows窗口的opencv
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    # 把windows窗口靠在一起
    cv2.moveWindow("base-image", 0, 100)
    cv2.moveWindow("result-image", 400, 100)

    # 启动我们正在使用的两个windows窗口的windows窗口线程
    cv2.startWindowThread()

    # 创建我们将使用的dlib的跟踪器
    tracker = dlib.correlation_tracker()

    # 我们用来跟踪当前是否使用dlib跟踪器的变量
    trackingFace = 0

    # 我们在脸周围画的矩形的颜色
    rectangleColor = (0, 165, 255)

    try:
        while True:
            # 从网络摄像头中检索最新图像
            rc, fullSizeBaseImage = capture.read()

            # 将图像大小调整为320x240
            baseImage = cv2.resize(fullSizeBaseImage, (320, 240))

            # 检查是否按下了某个键，如果是Q，则销毁所有opencv窗口并退出应用程序
            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('Q'):
                cv2.destroyAllWindows()
                exit(0)

            # 结果图像是我们将向用户显示的图像，它是来自网络摄像头的原始图像和最大人脸的叠加矩形的组合
            resultImage = baseImage.copy()

            # 如果我们没有跟踪一张脸，那就试着检测一张
            if not trackingFace:

                # 对于人脸检测，我们需要利用一个灰色的图像，这样我们就可以将基础图像转换为基于灰色的图像
                gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
                # 现在使用haar级联检测器查找图像中的所有面
                faces = faceCascade.detectMultiScale(gray, 1.3, 5)

                # 在控制台中，我们可以显示，只有现在我们才使用面部探测器
                print("Using the cascade detector to detect face")

                # 目前，我们只对“最大的”面感兴趣，我们根据找到的矩形的最大面积来确定。首先将所需变量初始化为0
                maxArea = 0
                x = 0
                y = 0
                w = 0
                h = 0

                # 在所有面上循环，并检查此面的面积是否是迄今为止最大的。
                # 由于dlib跟踪器的要求，我们需要在这里把它转换成int。
                # 如果我们在这里省略对int的强制转换，您将得到强制转换错误，因为检测器返回numpy.int32，而跟踪器需要一个int
                for (_x, _y, _w, _h) in faces:
                    if _w * _h > maxArea:
                        x = int(_x)
                        y = int(_y)
                        w = int(_w)
                        h = int(_h)
                        maxArea = w * h

                # 如果找到一个或多个面，请在图片中最大的面上初始化跟踪器
                if maxArea > 0:
                    # 初始化跟踪器
                    tracker.start_track(baseImage,
                                        dlib.rectangle(x - 10,
                                                       y - 20,
                                                       x + w + 10,
                                                       y + h + 20))
                    # 设置指示器变量，以便我们知道跟踪器正在跟踪图像中的某个区域
                    trackingFace = 1

            # 检查跟踪器是否正在主动跟踪图像中的某个区域
            if trackingFace:
                # 更新跟踪程序并请求有关跟踪更新质量的信息
                trackingQuality = tracker.update(baseImage)
                # 如果跟踪质量足够好，确定跟踪区域的更新位置并绘制矩形
                if trackingQuality >= 8.75:
                    tracked_position = tracker.get_position()
                    t_x = int(tracked_position.left())
                    t_y = int(tracked_position.top())
                    t_w = int(tracked_position.width())
                    t_h = int(tracked_position.height())
                    cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 2)

                else:
                    # 如果跟踪更新的质量不够（例如，被跟踪区域移出屏幕），我们将停止对人脸的跟踪，
                    # 在下一个循环中，我们将再次在图像中找到最大的人脸
                    trackingFace = 0

            # 因为我们想在屏幕上显示比原来的320x240更大的东西，所以我们再次调整图像的大小
            # 请注意，也可以保留大版本的基本图像，并使结果图像成为此大基本图像的副本，并使用缩放因子在右坐标处绘制矩形。
            largeResult = cv2.resize(resultImage, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))
            # 最后，我们想在屏幕上显示图像
            cv2.imshow("base-image", baseImage)
            cv2.imshow("result-image", largeResult)

    # 为了确保我们也可以处理用户在控制台中按Ctrl-C
    # 我们必须检查键盘中断异常并销毁所有opencv窗口并退出应用程序
    except KeyboardInterrupt as e:
        cv2.destroyAllWindows()
        exit(0)


if __name__ == '__main__':
    detectAndTrackLargestFace()
