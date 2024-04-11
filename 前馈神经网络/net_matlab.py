import numpy as np
import matplotlib.pyplot as plt


def F(x):
    return 1 / (1 + np.exp(-x))


def dF(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


def normalize(data):  # 归一化到（0，1）

    # 将输入转换为NumPy数组
    arr = np.array(data)

    # 计算最大值和最小值
    min_val = np.min(arr)
    max_val = np.max(arr)

    # 归一化
    normalized_data = (arr - min_val) / (max_val - min_val)

    return normalized_data


lr = 0.1
lmd = 0
random_x = np.random.uniform(-100, 100, 100).reshape(1, -1)
random_y = random_x**2
a0 = normalize(random_x)
y = normalize(random_y)

# a0 = np.random.uniform(-1, 1, 100).reshape(1, -1)
# y = a0**2 + np.random.rand(np.size(a0)) * 0.1

W1 = np.random.randn(10, 1)
W2 = np.random.randn(1, 10)
b1 = 1
b2 = 1

plt.subplot(1, 2, 1)
plt.scatter(a0, y)

for i in range(200):
    z1 = W1 @ a0 + b1  # 改变a0的形状以匹配W1的列数
    a1 = F(z1)
    z2 = W2 @ a1 + b2
    a2 = F(z2)
    if i == 199:
        plt.subplot(1, 2, 2)
        plt.scatter(a0, a2)

    derivatives2 = -2 * (y - a2) * dF(z2)
    derivatives1 = dF(z1) * np.dot(W2.T, derivatives2)

    dL_dW2 = derivatives2 @ a1.T / 100
    dL_dW1 = derivatives1 @ a0.T / 100

    W2 = W2 - lr * dL_dW2 - lr * lmd * W2
    W1 = W1 - lr * dL_dW1 - lr * lmd * W1
    b1 = b1 - lr * derivatives1
    b2 = b2 - lr * derivatives2

plt.show()
plt.savefig("matlab_net.png")
