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
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)["out"]
    output_predictions = output.argmax(1).cpu().numpy()
    mask = (output_predictions == 0)  # 假设背景类为0
    mask = np.repeat(mask[:, :, :, np.newaxis], 3, axis=3)

    image_np = np.array(image)
    image_np = cv2.resize(image_np, (224, 224))

    image_no_bg = image_np * mask
    image_no_bg = Image.fromarray(image_no_bg.astype(np.uint8))

    return image_no_bg


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
