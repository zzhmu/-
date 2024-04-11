import cv2
import numpy as np
import os


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # 加载图像
    img = cv2.imread(image_path)
    # 将BGR格式转换为RGB格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 调整图像大小
    img = cv2.resize(img, target_size)
    # 将像素值缩放到0到1之间
    img = img.astype(np.float32) / 255.0
    # 添加批处理维度
    img = np.expand_dims(img, axis=0)
    return img


def load_images_from_folder(folder_path, target_size=(224, 224)):
    image_tensors = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 构建图像文件的完整路径
        img_path = os.path.join(folder_path, filename)
        # 检查文件是否是图像文件
        if os.path.isfile(img_path) and any(
            img_path.endswith(extension)
            for extension in [".jpg", ".jpeg", ".png", ".bmp"]
        ):
            # 加载并预处理图像
            img_tensor = load_and_preprocess_image(img_path, target_size)
            image_tensors.append(img_tensor)
    return np.concatenate(image_tensors, axis=0)


# 图像文件夹路径
folder_path = "image_folder"  # 你的图像文件夹路径
target_size = (224, 224)  # 目标图像大小

# 加载并预处理图像
image_tensor = load_images_from_folder(folder_path, target_size)

print("Total number of images loaded:", image_tensor.shape[0])
