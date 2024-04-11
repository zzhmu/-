import numpy as np
import matplotlib.pyplot as plt


class MakeSet:  # 创建数据集
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


class CreateNeuronNet:  # 创建一层
    def __init__(self, thisL, lastL):

        inital_bias = 1
        self.W = np.random.randn(thisL, lastL)
        self.b = np.full((thisL, 1), inital_bias)


def normalize(data):  # 归一化到（0，1）

    # 将输入转换为NumPy数组
    arr = np.array(data)

    # 计算最大值和最小值
    min_val = np.min(arr)
    max_val = np.max(arr)

    # 归一化
    normalized_data = (arr - min_val) / (max_val - min_val)

    return normalized_data


def TrainingNet(trainX, trainY, alpha, lambda_):  # 训练
    x = trainX
    y = trainY
    a0 = x
    z1 = layer1.W @ a0 + layer1.b
    a1 = Sigmoid(z1)
    z2 = layer2.W @ a1 + layer2.b
    a2 = Sigmoid(z2)
    delta2 = 2 * dQuadraticLoss(a2, y) @ dSigmoid(z2)  # 输出层误差
    delta1 = dSigmoid(z1) * (layer2.W.T @ delta2)  # 隐藏层误差

    layer2.W = layer2.W - alpha * (delta2 @ a1.T + lambda_ * layer2.W)
    layer2.b = layer2.b - alpha * delta2
    layer1.W = layer1.W - alpha * (delta1 @ a0.T + lambda_ * layer1.W)
    layer1.b = layer1.b - alpha * delta1


def UseNet(test_X):  # 使用网络计算

    a0 = test_X
    z1 = layer1.W @ a0 + layer1.b
    a1 = Sigmoid(z1)
    z2 = layer2.W @ a1 + layer2.b
    a2 = Sigmoid(z2)
    test_Y = a2
    return test_Y


def ReLU(x):
    return np.maximum(0, x)


def Linear(x):
    return x


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dSigmoid(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


def dQuadraticLoss(a, y):
    return a - y


random_x = np.random.uniform(-100, 100, 100).reshape(1, -1)
random_y = random_x**2
random_x = normalize(random_x)
random_y = normalize(random_y)

testX = np.random.uniform(-100, 100, 100).reshape(1, -1)
testY = testX**2
testX = normalize(testX)
testY = normalize(testY)

layer1 = CreateNeuronNet(18, 100)
layer2 = CreateNeuronNet(100, 18)

testSet = MakeSet(testX, testY)
trainingSet = MakeSet(random_x, random_y)


alpha = 0.01
lambda_ = 0

for i in range(10):
    TrainingNet(trainingSet.X, trainingSet.Y, alpha, lambda_)


plt.scatter(testSet.X, testSet.Y, color="blue", label="TrainingSet", s=1)
plt.scatter(testSet.X, UseNet(testSet.X), color="red", label="TrainingResult", s=1)
# 设置图表标题和坐标轴标签
plt.title("TrainingResult")
plt.show()
plt.savefig("result.png")
