import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import shutil


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


if __name__ == "__main__":
    # 读取pre文件夹中的所有图像，处理并保存到input文件夹
    input_dir = "pre"
    output_dir = "input"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            image = Image.open(image_path).convert("RGB")

            image_no_bg = remove_background(image, kernel_size=3, morph_type=cv2.MORPH_OPEN)

            if is_white_background(image, image_no_bg, similarity_threshold=0.9):
                # 如果背景是白色，将文件复制到output文件夹并打印消息
                shutil.copy(image_path, output_path)
                print(f"{file_name} 为背景白色，跳过处理")
            else:
                # 否则，保存处理后的图像
                image_no_bg.save(output_path)

# if __name__ == "__main__":
#     # 读取pre文件夹中的所有图像，处理并保存到input文件夹
#     input_dir = "pre"
#     output_dir = "input"
#
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # kernel_sizes = list(range(3, 16, 2))
#     # morph_types = {
#     #     'OPEN': cv2.MORPH_OPEN,
#     #     'DILATE': cv2.MORPH_DILATE,
#     #     'ERODE': cv2.MORPH_ERODE
#     # }
#     #
#     # for file_name in os.listdir(input_dir):
#     #     if file_name.endswith((".jpg", ".jpeg", ".png", ".bmp")):
#     #         image_path = os.path.join(input_dir, file_name)
#     #
#     #         # 读取图像并转换为RGB
#     #         image = Image.open(image_path).convert("RGB")
#     #
#     #         for kernel_size in kernel_sizes:
#     #             for morph_type_name, morph_type in morph_types.items():
#     #                 output_file_name = f"{os.path.splitext(file_name)[0]}_kernel{kernel_size}_morph{morph_type_name}{os.path.splitext(file_name)[1]}"
#     #                 output_path = os.path.join(output_dir, output_file_name)
#     #
#     #                 image_no_bg = remove_background(image, kernel_size=kernel_size, morph_type=morph_type)
#     #                 image_no_bg.save(output_path)
#
#     for file_name in os.listdir(input_dir):
#         if file_name.endswith((".jpg", ".jpeg", ".png", ".bmp")):
#             image_path = os.path.join(input_dir, file_name)
#             output_path = os.path.join(output_dir, file_name)
#             image = Image.open(image_path).convert("RGB")
#             # image_no_bg = remove_background(image)
#             # 内核大小（kernel_size）：这是形态学操作中使用的结构元素的大小。较大的内核将产生更平滑的边缘，但可能导致形状失真。建议在3到15之间选择一个奇数值。
#             # 形态学操作类型（morph_type）： 如开运算（cv2.MORPH_OPEN）、膨胀（cv2.MORPH_DILATE）或腐蚀（cv2.MORPH_ERODE）以查看是否有更好的效果。
#             # image_no_bg = remove_background(image, kernel_size=7, morph_type=cv2.MORPH_CLOSE)
#             image_no_bg = remove_background(image, kernel_size=3, morph_type=cv2.MORPH_OPEN)
#             image_no_bg.save(output_path)
