import os
import re
import warnings
import scipy.misc
import cv2
import face_recognition
from PIL import Image
import argparse
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images_dir", help="image dir")
parser.add_argument("-v", "--video", help="video to recognize faces on")
parser.add_argument("-o", "--output_csv", help="Ouput csv file [Optional]")
parser.add_argument("-u", "--upsample-rate",
                    help="How many times to upsample the image looking for faces. Higher numbers find smaller faces. [Optional]")
args = vars(parser.parse_args())

# 检查参数值是否有效
if args.get("images_dir", None) is None and os.path.exists(str(args.get("images_dir", ""))):
    print("Please check the path to images folder")
    exit()
if args.get("video", None) is None and os.path.isfile(str(args.get("video", None))):
    print("Please check the path to video")
    exit()
if str(args.get("output_csv", None)) is None:
    print("You haven't specified an output csv file. Nothing will be written.")
# 默认情况下，upsample rate(上采样率)=1
upsample_rate = args.get("upsample_rate", None)
if upsample_rate is None:
    upsample_rate = 1


# 辅助函数
def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(pgm|jpg|png)', f, flags=re.I)]


def test_image(image_to_check, known_names, known_face_encodings, number_of_times_to_upsample=1):
    """
    通过检查已知图像来检测未知图像中是否有人脸被识别
    :param image_to_check: 图像的Numpy数组
    :param known_names: 包含已知标签的列表
    :param known_face_encodings: 包含训练图像标签的列表
    :param number_of_times_to_upsample: 多少次对图像进行重新采样以查找面。数字越高，脸越小。
    :return: 已知名称的标签列表
    """
    # 未知图像unknown_image = face_recognition.load_image_file(image_to_check)
    unknown_image = image_to_check
    # 缩小图像以使其运行更快
    if unknown_image.shape[1] > 1600:
        scale_factor = 1600 / unknown_image.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        unknown_image = scipy.misc.imresize(unknown_image, scale_factor)
    face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample)
    unknown_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    result = []
    for unknown_encoding in unknown_encodings:
        result = face_recognition.compare_faces(known_face_encodings, unknown_encoding)

    result_encoding = []
    for nameIndex, is_match in enumerate(result):
        if is_match:
            result_encoding.append(known_names[nameIndex])

    return result_encoding


def map_file_pattern_to_label(labels_with_pattern, labels_list):  # result
    """
    将文件名模式映射到完整标签
    :param labels_with_pattern: dict : { "file_name_pattern": "full_label" }
    :param labels_list: list : 从test_image()获取的文件名标签列表
    :return: 完整标签列表
    """
    result_list = []
    for key, label in labels_with_pattern.items():
        for img_labels in labels_list:
            if str(key).lower() in str(img_labels).lower():
                if str(label) not in result_list:
                    result_list.append(str(label))
                # continue
    # result_list = [label for key, label in labels_with_pattern if str(key).lower() in labels_list]
    return result_list


cap = cv2.VideoCapture(args["video"])

# 获取训练图像
training_encodings = []
training_labels = []
for file in image_files_in_folder(str(args['images_dir'])):
    basename = os.path.splitext(os.path.basename(file))[0]
    img = face_recognition.load_image_file(file)
    encodings = face_recognition.face_encodings(img)

    if len(encodings) > 1:
        print("WARNING: More than one face found in {}. Only considering the first face.".format(file))

    if len(encodings) == 0:
        print("WARNING: No faces found in {}. Ignoring file.".format(file))
    if len(encodings):
        training_labels.append(basename)
        training_encodings.append(encodings[0])

csvfile = None
csvwriter = None
if args.get("output_csv", None) is not None:
    csvfile = open(args.get("output_csv"), 'w')
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

ret, firstFrame = cap.read()
frameRate = cap.get(cv2.CAP_PROP_FPS)

# 带有文件模式的标签，编辑此处
label_pattern = {
    "pooja": "Shahrukh Khan", "j": "Ameer Khan"
}

# 将视频中的每一帧与我们训练过的一组标记图像相匹配
while ret:
    curr_frame = cap.get(1)
    ret, frame = cap.read()
    result = test_image(frame, training_labels, training_encodings, upsample_rate)
    print(result)
    labels = map_file_pattern_to_label(label_pattern, result)
    print(labels)
    curr_time = curr_frame / frameRate
    print("Time: {} faces: {}".format(curr_time, labels))
    if csvwriter:
        csvwriter.writerow([curr_time, labels])
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
if csvfile:
    csvfile.close()
cap.release()
cv2.destroyAllWindows()
