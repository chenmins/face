import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

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


def remove_background(image):
    original_size = image.size

    # 将图像缩放到模型输入尺寸
    resized_image = image.resize((224, 224), Image.ANTIALIAS)

    input_tensor = preprocess(resized_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)["out"]
    output_predictions = output.argmax(1).cpu().numpy()

    mask = (output_predictions != 0)  # 将不等于背景类的像素保留为主体
    mask = np.repeat(mask[:, :, :, np.newaxis], 3, axis=3)

    image_np = np.array(resized_image)
    image_no_bg = image_np * mask[0]  # 选择第一个元素，以获取正确的形状

    # 将背景设置为白色
    background = np.ones_like(image_no_bg) * 255
    background[np.where(mask[0] == 1)] = 0

    image_white_bg = image_no_bg + background
    image_white_bg = Image.fromarray(image_white_bg.astype(np.uint8))

    # 使用原始掩码将图像恢复到原始尺寸
    mask_original_size = np.array(Image.fromarray(mask[0].astype(np.uint8)).resize(original_size, Image.ANTIALIAS))
    image_white_bg_original_size = Image.fromarray(np.array(image) * mask_original_size + np.array(Image.new('RGB', original_size, (255, 255, 255))) * (1 - mask_original_size))

    return image_white_bg_original_size


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
            image_no_bg = remove_background(image)
            image_no_bg.save(output_path)
