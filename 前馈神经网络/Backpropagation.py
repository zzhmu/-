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
    number = 0
    d2 = 0
    d1 = 0
    for x, y in np.nditer([trainX, trainY]):
        number = number + 1
        a0 = np.array([[x]])
        z1 = layer1.W @ a0 + layer1.b
        a1 = Sigmoid(z1)
        z2 = layer2.W @ a1 + layer2.b
        a2 = Sigmoid(z2)
        delta2 = dQuadraticLoss(a2, y) * dSigmoid(z2)  # 输出层误差
        delta1 = dSigmoid(z1) * (layer2.W.T @ delta2)  # 隐藏层误差
        d2 = d2 + delta2
        d1 = d1 + delta1
        if number == 50:
            layer2.W = layer2.W - alpha * (d2 / number @ a1.T + lambda_ * layer2.W)
            layer2.b = layer2.b - alpha * d2 / number
            layer1.W = layer1.W - alpha * (d1 / number @ a0.T + lambda_ * layer1.W)
            layer1.b = layer1.b - alpha * d1 / number

            d2 = d2 - d2
            d1 = d1 - d1
            number = 0


def UseNet(test_X):  # 使用网络计算
    # a0 = test_X
    # z1 = layer1.W @ a0 + layer1.b
    # a1 = Sigmoid(z1)
    # z2 = layer2.W @ a1 + layer2.b
    # a2 = Sigmoid(z2)
    # return a2
    test_Y = np.zeros_like(test_X)
    for i, x in enumerate(test_X):
        a0 = np.array([[x]])
        z1 = layer1.W @ a0 + layer1.b
        a1 = Sigmoid(z1)
        z2 = layer2.W @ a1 + layer2.b
        a2 = Sigmoid(z2)
        test_Y[i] = a2[0, 0]
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


testX = np.random.uniform(-1, 1, 50).reshape(1, -1)
testY = testX**2
# testX = normalize(testX)
# testY = normalize(testY)

layer1 = CreateNeuronNet(10, 1)
layer2 = CreateNeuronNet(1, 10)

testSet = MakeSet(testX, testY)


alpha = 0.1
lambda_ = 0

for i in range(200000):
    random_x = np.random.uniform(-1, 1, 50).reshape(1, -1)
    random_y = random_x**2
    # random_x = normalize(random_x)
    # random_y = normalize(random_y)
    trainingSet = MakeSet(random_x, random_y)
    TrainingNet(trainingSet.X, trainingSet.Y, alpha, lambda_)

initial_W1 = layer1.W
initial_W2 = layer2.W

# for i in range(40000):
#     train_x = np.random.uniform(-1, 1, 50).reshape(1, -1)
#     train_y = train_x**2
#     trainingSet = MakeSet(train_x, train_y)
#     x = trainingSet.X
#     y = trainingSet.Y
#     a0 = x
#     z1 = layer1.W @ a0 + layer1.b
#     a1 = Sigmoid(z1)
#     z2 = layer2.W @ a1 + layer2.b
#     a2 = Sigmoid(z2)
#     # if i == 199:
#     #     plt.subplot(1, 2, 2)
#     #     plt.scatter(a0, a2)
#     #     # plt.scatter(
#     #     #     trainingSet.X,
#     #     #     UseNet(trainingSet.X),
#     #     #     color="red",
#     #     #     label="TrainingResult",
#     #     #     s=1,
#     #     # )
#     delta2 = dQuadraticLoss(a2, y) * dSigmoid(z2)  # 输出层误差
#     delta1 = dSigmoid(z1) * (layer2.W.T @ delta2)  # 隐藏层误差

#     layer2.W = layer2.W - alpha * (
#         delta2 @ a1.T / 100 + lambda_ * layer2.W
#     )  # 为什么只有在这儿除100才行
#     layer2.b = layer2.b - alpha * delta2
#     layer1.W = layer1.W - alpha * (delta1 @ a0.T / 100 + lambda_ * layer1.W)
#     layer1.b = layer1.b - alpha * delta1

# plt.subplot(1, 4, 1)
# plt.scatter(trainingSet.X, trainingSet.Y, color="blue", label="TrainingSet", s=1)
# plt.title("TrainingSet")
# # plt.subplot(1, 4, 2)
# plt.scatter(
#     trainingSet.X, UseNet(trainingSet.X), color="red", label="TrainingResult", s=1
# )
# plt.title("Result-TrainSet")
plt.subplot(1, 4, 3)
plt.scatter(testSet.X, testSet.Y, color="blue", label="TrainingResult", s=1)
plt.title("Result-TestSet")
# plt.subplot(1, 4, 4)
plt.scatter(
    testSet.X, UseNet(testSet.X), color="red", label="TrainingResult", s=1
)  # 因为只对训练集训练了100遍，所以只拟合训练集
plt.title("Result-TestSet")
# 设置图表标题和坐标轴标签

plt.show()
plt.savefig("result.png")
