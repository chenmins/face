import os
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from urllib.request import urlretrieve
from flask import Flask, request, send_file
from mtcnn.mtcnn import MTCNN
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from werkzeug.utils import secure_filename
import functools
import hashlib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet50(pretrained=True).to(device)
model.eval()

# 定义预处理和后处理方法
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ... 保留 remove_background, is_white_background, extract_foreground, replace_background_with_white 和 crop_face 的实现 ...


def remove_background(image, kernel_size=5, morph_type=cv2.MORPH_CLOSE):
    original_size = image.size

    # 将图像缩放到模型输入尺寸
    resized_image = image.resize((224, 224), Image.ANTIALIAS)

    input_tensor = preprocess(resized_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)["out"]
    output_predictions = output.argmax(1).cpu().numpy()

    mask = (output_predictions != 0)  # 将不等于背景类的像素保留为主体
    mask = np.repeat(mask[:, :, :, np.newaxis], 3, axis=3)

    # 对生成的掩码进行形态学操作以平滑边缘
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_smooth = cv2.morphologyEx(mask[0].astype(np.uint8), morph_type, kernel)

    image_np = np.array(resized_image)
    image_no_bg = image_np * mask_smooth

    # 将背景设置为白色
    background = np.ones_like(image_no_bg) * 255
    background[np.where(mask_smooth == 1)] = 0

    image_white_bg = image_no_bg + background
    image_white_bg = Image.fromarray(image_white_bg.astype(np.uint8))

    # 使用原始掩码将图像恢复到原始尺寸
    mask_original_size = np.array(Image.fromarray(mask_smooth).resize(original_size, Image.ANTIALIAS))
    image_white_bg_original_size = Image.fromarray(
        np.array(image) * mask_original_size + np.array(Image.new('RGB', original_size, (255, 255, 255))) * (
                1 - mask_original_size))

    return image_white_bg_original_size


def is_white_background(image, image_no_bg, similarity_threshold=0.99):
    # 将两个图像转换为 NumPy 数组
    image_np = np.array(image)
    image_no_bg_np = np.array(image_no_bg)

    # 计算原始图像和去除背景后的图像之间的相似度
    similarity = np.sum(image_np == image_no_bg_np) / image_np.size

    # 如果相似度大于等于给定阈值，则认为背景是白色的
    return similarity >= similarity_threshold


def extract_foreground(img, iterations=5):
    mask = np.zeros(img.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    rect = (0, 0, img.shape[1] - 1, img.shape[0] - 1)

    cv2.grabCut(img, mask, rect, bg_model, fg_model, iterations, mode=cv2.GC_INIT_WITH_RECT)
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

    return mask_binary


def replace_background_with_white(img, mask_binary):
    white_bg = np.full(img.shape, 255, dtype=np.uint8)
    img_foreground = cv2.bitwise_and(img, img, mask=mask_binary)
    img_background = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(mask_binary))
    result = cv2.add(img_foreground, img_background)
    return result


def face(input_dir, output_dir, img_size=300):
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


def crop_face(input_dir, output_dir, expand_factor=1.8, img_size=300):
    detector = MTCNN()

    file = input_dir
    input_file = input_dir
    img = cv2.imread(input_file)
    faces = detector.detect_faces(img)

    if len(faces) > 0:
        face = max(faces, key=lambda face: face['box'][2] * face['box'][3])  # 选择最大的人脸
        x, y, width, height = face['box']
        center_x, center_y = x + width // 2, y + height // 2

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

        # 创建一个与目标图像大小相同的白色背景
        padded_face = np.full((img_size, img_size, 3), 255, dtype=np.uint8)

        # 将调整后的头像粘贴到背景中央
        padded_face[(img_size - new_height) // 2: (img_size - new_height) // 2 + new_height,
        (img_size - new_width) // 2: (img_size - new_width) // 2 + new_width] = resized_face

        # 创建一个与目标图像大小相同的透明遮罩
        mask = np.zeros((img_size, img_size, 4), dtype=np.uint8)

        # 画内切圆
        cv2.circle(mask, (img_size // 2, img_size // 2), img_size // 2, (255, 255, 255, 255), -1)

        # 将原图转为带透明通道的图像
        padded_face_rgba = cv2.cvtColor(padded_face, cv2.COLOR_BGR2BGRA)

        # 合并原图和遮罩
        masked_face = cv2.bitwise_and(padded_face_rgba, mask)
        cv2.imwrite(output_dir, masked_face)


cache = {}


def process_image(input_path, output_path, expand_factor, img_size, need_crop):
    if os.path.exists(output_path):
        return send_file(output_path, mimetype='image/png')
    else:
        image = Image.open(input_path).convert("RGB")
        image_no_bg = remove_background(image, kernel_size=3, morph_type=cv2.MORPH_OPEN)
        image_no_bg.save(input_path)
        if need_crop:
            crop_face(input_path, output_path, expand_factor, img_size)
        else:
            face(input_path, output_path, expand_factor, img_size)
        cache[input_path] = output_path
        return send_file(output_path, mimetype='image/png')


def handle_request(input_path, expand_factor, img_size, need_crop=True):
    image = Image.open(input_path).convert("RGB")
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    output_filename = f"{image_hash}.png"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output", output_filename)

    if input_path in cache:
        return send_file(cache[input_path], mimetype='image/png')
    else:
        return process_image(input_path, output_path, expand_factor, img_size, need_crop)


@app.route('/api/ai/face_file', methods=['POST'])
def face_file():
    file = request.files.get('file')
    expand_factor = float(request.form.get('expand_factor', 1.8))
    img_size = int(request.form.get('img_size', 600))

    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        return handle_request(input_path, expand_factor, img_size, False)
    else:
        return "No file provided", 400


@app.route('/api/ai/crop_face_file', methods=['POST'])
def crop_face_file():
    file = request.files.get('file')
    expand_factor = float(request.form.get('expand_factor', 1.8))
    img_size = int(request.form.get('img_size', 300))

    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        return handle_request(input_path, expand_factor, img_size)
    else:
        return "No file provided", 400


@app.route('/api/ai/crop_face_url', methods=['GET'])
def crop_face_url():
    url = request.args.get('url')
    expand_factor = float(request.args.get('expand_factor', 1.8))
    img_size = int(request.args.get('img_size', 300))

    if url:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(url))
        urlretrieve(url, input_path)
        return handle_request(input_path, expand_factor, img_size)
    else:
        return "No URL provided", 400


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True
    )
