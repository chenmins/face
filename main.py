import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from sklearn.cluster import KMeans


def replace_dominant_color_with_white(img, n_clusters=3, percentage_threshold=0.5):
    # 调整图像大小以加快处理速度
    small_img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)

    # 将图像从 BGR 转为 RGB
    small_img_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    # 将图像数据重塑为二维数组
    pixel_data = small_img_rgb.reshape(-1, 3)

    # 使用 K-means 算法对像素数据进行聚类
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixel_data)
    cluster_centers = kmeans.cluster_centers_

    # 获取每个聚类的像素计数
    unique, counts = np.unique(kmeans.labels_, return_counts=True)

    # 计算每个聚类的像素占比
    percentages = counts / sum(counts)

    # 创建一个空白遮罩
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # 遍历每个聚类，检查其占比是否大于阈值
    for i, percentage in enumerate(percentages):
        if percentage > percentage_threshold:
            dominant_color = cluster_centers[i].astype(int)

            # 为大于阈值的颜色创建遮罩
            color_mask = cv2.inRange(img, dominant_color - 50, dominant_color + 50)
            mask = cv2.bitwise_or(mask, color_mask)

    mask_inv = cv2.bitwise_not(mask)

    # 创建一个白色背景
    white_img = np.full(img.shape, 255, dtype=np.uint8)

    # 使用遮罩将大于阈值的颜色替换为白色
    img_foreground = cv2.bitwise_and(img, img, mask=mask_inv)
    img_background = cv2.bitwise_and(white_img, white_img, mask=mask)

    result = cv2.add(img_foreground, img_background)
    return result




def replace_dominant_color_with_white66(img, n_clusters=5):
    small_img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    small_img_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
    pixel_data = small_img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixel_data)
    cluster_centers = kmeans.cluster_centers_
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color = cluster_centers[np.argmax(counts)].astype(int)
    mask = cv2.inRange(img, dominant_color - 30, dominant_color + 30)
    mask_inv = cv2.bitwise_not(mask)
    white_img = np.full(img.shape, 255, dtype=np.uint8)
    img_foreground = cv2.bitwise_and(img, img, mask=mask_inv)
    img_background = cv2.bitwise_and(white_img, white_img, mask=mask)
    result = cv2.add(img_foreground, img_background)
    return result


def replace_dominant_color_with_white11(img, n_clusters=3):
    # 调整图像大小以加快处理速度
    small_img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)

    # 将图像从 BGR 转为 RGB
    small_img_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    # 将图像数据重塑为二维数组
    pixel_data = small_img_rgb.reshape(-1, 3)

    # 使用 K-means 算法对像素数据进行聚类
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixel_data)
    cluster_centers = kmeans.cluster_centers_

    # 获取最常见的颜色
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color = cluster_centers[np.argmax(counts)].astype(int)

    # 为最常见的颜色创建一个遮罩
    mask = cv2.inRange(img, dominant_color - 30, dominant_color + 30)
    mask_inv = cv2.bitwise_not(mask)

    # 创建一个白色背景
    white_img = np.full(img.shape, 255, dtype=np.uint8)

    # 使用遮罩将最常见的颜色替换为白色
    img_foreground = cv2.bitwise_and(img, img, mask=mask_inv)
    img_background = cv2.bitwise_and(white_img, white_img, mask=mask)

    result = cv2.add(img_foreground, img_background)
    return result


def replace_non_white_with_white(img, threshold=230):
    lower_bound = np.array([threshold, threshold, threshold])
    upper_bound = np.array([255, 255, 255])

    white_mask = cv2.inRange(img, lower_bound, upper_bound)
    white_mask_inv = cv2.bitwise_not(white_mask)

    white_img = np.full(img.shape, 255, dtype=np.uint8)
    img_foreground = cv2.bitwise_and(img, img, mask=white_mask_inv)
    img_background = cv2.bitwise_and(white_img, white_img, mask=white_mask)

    result = cv2.add(img_foreground, img_background)
    return result


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


def crop_face(input_dir, output_dir, img_size=300):
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



                # 提取前景并与白色背景合并
                # mask_binary = extract_foreground(cropped_face, iterations=5)
                # cropped_face = replace_background_with_white(cropped_face, mask_binary)

                # 将非白色背景替换为白色背景
                # cropped_face = replace_non_white_with_white(cropped_face, threshold=240)

                # 将最常见的背景颜色替换为白色
                # cropped_face = replace_dominant_color_with_white(cropped_face, n_clusters=5)
                cropped_face = replace_dominant_color_with_white(cropped_face, n_clusters=10, percentage_threshold=0.44)

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

                output_file = os.path.join(output_dir, os.path.splitext(file)[0] + '.png')
                cv2.imwrite(output_file, masked_face)


if __name__ == "__main__":
    input_dir = "input/"  # 输入文件夹路径，包含原始头像图片
    output_dir = "output/"  # 输出文件夹路径，用于保存裁剪后的头像
    crop_face(input_dir, output_dir)
