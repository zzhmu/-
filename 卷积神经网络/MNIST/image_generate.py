import numpy as np
import os
from PIL import Image


def read_mnist_images(filename):
    with open(filename, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        return images


def read_mnist_labels(filename):
    with open(filename, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        num_labels = int.from_bytes(f.read(4), "big")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


def save_images(images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, img in enumerate(images):
        img = Image.fromarray(img)
        img.save(os.path.join(output_dir, f"image_{i}.png"))


# 读取MNIST图像和标签数据
mnist_images = read_mnist_images(
    r"U:\Git-Repositories\neuron_net\卷积神经网络\MNIST\train-images.idx3-ubyte"
)
mnist_labels = read_mnist_labels(
    r"U:\Git-Repositories\neuron_net\卷积神经网络\MNIST\train-labels.idx1-ubyte"
)


# # 将图像与标签对应起来
# for i in range(len(mnist_labels)):
#     img = mnist_images[i]
#     label = mnist_labels[i]
#     # 在这里可以进行进一步的处理，例如显示图像和标签
#     print(f"Image {i}: Label = {label}")

# # 保存图像和标签
# np.save("mnist_images.npy", mnist_images)
# np.save("mnist_labels.npy", mnist_labels)
# print("MNIST图像和标签已保存为 NumPy 数组文件。")

# 创建保存图像的文件夹
output_folder = r"U:\Git-Repositories\neuron_net\卷积神经网络\MNIST\mnist_images"

# 指定图像保存的文件夹路径
image_output_folder = r"U:\Git-Repositories\neuron_net\卷积神经网络\MNIST\mnist_images"
# 指定标签保存的文件夹路径
label_output_folder = r"U:\Git-Repositories\neuron_net\卷积神经网络\MNIST\mnist_labels"

# 创建图像保存文件夹
os.makedirs(image_output_folder, exist_ok=True)
# 创建标签保存文件夹
os.makedirs(label_output_folder, exist_ok=True)
num_images_to_save = 100
images_saved = 0
# 保存 MNIST 图像和标签为文件
for i, (image, label) in enumerate(zip(mnist_images, mnist_labels)):
    if images_saved >= num_images_to_save:
        break
    # 保存图像
    image_path = os.path.join(image_output_folder, f"{i}.png")
    image = Image.fromarray(image)
    image.save(image_path)

    # 保存图像对应的标签到文件
    with open(os.path.join(label_output_folder, f"{i}_label.txt"), "w") as f:
        f.write(str(label))

    images_saved += 1
