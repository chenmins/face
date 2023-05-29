import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN


def face(input_dir, output_dir, img_size=600):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detector = MTCNN()
    for root, _, files in os.walk(input_dir):
        for file in files:
            input_file = os.path.join(root, file)
            img = cv2.imread(input_file)
            faces = detector.detect_faces(img)
            print(len(faces))

            if len(faces) > 0:
                face = max(faces, key=lambda face: face['box'][2] * face['box'][3])  # 选择最大的人脸
                x, y, width, height = face['box']
                center_x, center_y = x + width // 2, y + height // 2

                expand_factor = 1.8  # 根据需要调整扩展因子，这里取1.2
                half_size = int(max(width, height) * expand_factor) // 2

                start_x, end_x = max(0, center_x - half_size), min(img.shape[1], center_x + half_size)
                start_y, end_y = max(0, center_y - half_size), min(img.shape[0], center_y + half_size)

                cropped_face = img[start_y:end_y, start_x:end_x]

                # 计算等比例缩放因子
                scale_factor = img_size / max(cropped_face.shape[0], cropped_face.shape[1])
                new_width = int(cropped_face.shape[1] * scale_factor)
                new_height = int(cropped_face.shape[0] * scale_factor)

                # 使用等比例缩放调整头像大小
                resized_face = cv2.resize(cropped_face, (new_width, new_height), interpolation=cv2.INTER_AREA)

                output_file = os.path.join(output_dir, os.path.splitext(file)[0] + '.jpg')
                cv2.imwrite(output_file, resized_face)


if __name__ == "__main__":
    input_dir = "input/"  # 输入文件夹路径，包含原始头像图片
    output_dir = "output/"  # 输出文件夹路径，用于保存裁剪后的头像
    face(input_dir, output_dir)
