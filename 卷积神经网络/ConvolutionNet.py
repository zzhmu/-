import numpy as np

# import image_load


def CrossEntropyLossFunction(y):
    return -np.log(y)


def ZeroPadding(data, pad):  # 零填充
    X_pad = np.pad(
        data,
        ((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return X_pad


def FullPadding():
    return


def NarrowConvolution():
    return


def WideConvolution():
    return


def Equal_WidthConvolution():
    return


class ConvolutionLayer:  # 卷积层
    def __init__(
        self, data, coreSize=(3, 3), coreNumber=1, dorpout=False
    ):  # 卷积层默认尺度
        M, N, D = data.shape
        self.M = M
        self.N = N
        self.D = D
        self.data = data
        self.size = coreSize
        self.W = np.random.randn(*coreSize, self.D, coreNumber)
        self.b = 0
        self.coreNumber = coreNumber
        self.dropout = dorpout

    def _Convolution(self, Stride=1, Padding=0):
        # data = ZeroPadding(data, Padding)

        m, n, D, P = self.W.shape
        M_par, N_par = (self.M - m) / Stride + 1, (
            self.N - n
        ) / Stride + 1  # 在不同维度循环的次数,也就是经过卷积后的图片边长
        M_par = int(M_par)
        N_par = int(N_par)
        Z = np.zeros(
            (M_par, N_par, self.coreNumber)
        )  # 初始化输入经过权重之后的结果，经过卷积后的方块大小
        for p in range(self.coreNumber):
            for i in range(M_par):
                for j in range(N_par):
                    Z[i, j, p] += self.b  # 对输出矩阵的某个点先加上偏置
                    for u in range(m):  # 进行单个核的卷积操作
                        for v in range(n):
                            for d in range(self.D):
                                cW = self.W[:, :, d, p]
                                # layer[:, :, np.newaxis, np.newaxis] 这个可以加一个维度
                                Z[i, j, p] += (
                                    cW[u, v]
                                    * self.data[
                                        Stride * i + u - Stride,
                                        Stride * j + v - Stride,
                                        d,
                                    ]  # data也是一个矩阵
                                )
        Y = ReLU(Z)
        return Y


class PoolingLayer:  # 汇聚层
    def __init__(self, data, coreSize=(2, 2)):
        self.data = data
        M, N, D = data.shape
        self.M = M
        self.N = N
        self.D = D
        self.size = coreSize
        self.W = np.random.randn(*coreSize, D, 1)  # 汇聚层只有一个核
        self.b = 0

    def _pooling(self, Stride=2, Padding=0):
        m, n = self.size
        M_par, N_par = (self.M - m) / Stride + 1, (
            self.N - n
        ) / Stride + 1  # 在不同维度循环的次数,也就是经过卷积后的图片边长
        M_par = int(M_par)
        N_par = int(N_par)
        Z = np.zeros(
            (M_par, N_par, self.D)
        )  # 初始化输入经过权重之后的结果，经过卷积后的方块大小
        if self.M % 2 == 1:  # 当图边长为奇数时舍去最后一条边
            self.M -= 1
        if self.N % 2 == 1:
            self.N -= 1
        for d in range(self.D):
            for i in range(M_par):  # 选择核上某个点
                for j in range(N_par):
                    for u in range(0, self.M, 2):  # 2是因为coreSize
                        for v in range(0, self.N, 2):
                            Z[i, j, d] = np.max(
                                [
                                    self.data[u, v, d],
                                    self.data[u + 1, v, d],
                                    self.data[u, v + 1, d],
                                    self.data[u + 1, v + 1, d],
                                ]
                            )  # 这里汇聚核是2*2，所以就这么简单写了一下代码
        # Y = Softmax(Z)
        return Z


class FullyConnectedLayer:  # 全连接层
    def __init__(self, data, outputSize):
        self.data = data
        self.dataSize = data.shape
        self.W = np.random.randn(self.dataSize[0], outputSize)
        self.b = 0
        self.outputSize = outputSize

    def Dense(self):
        Z = self.data @ self.W
        Y = Softmax(Z)
        return Y


class Net:
    def __init__(self):
        return

    def CreateConvolutionLayer(self, input_data, coreSize, coreNumber):
        return ConvolutionLayer(input_data, coreSize, coreNumber)

    def CreatePoolingLayer(self, input_data, coreSize):
        return PoolingLayer(input_data, coreSize)

    def CreateFullyConnectedLayer(self, input_data, outputSize):
        return FullyConnectedLayer(input_data, outputSize)

    def _forward_propagation(self, layer):  # 暂时先没用 TODO
        layer._Convolution()

    def __back_propagation():
        return


def ReLU(x):
    return np.maximum(0, x)


def Softmax(x):
    exp_values = np.exp(x)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities


input = np.zeros((28, 28))
tensor = np.random.randn(8, 8)
trainingX = [np.random.randn(28, 28, 1) for _ in range(5)]
trainingY = [1, 0, 1, 0, 1]


class TrainingSet:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


trainingSet = TrainingSet(trainingX, trainingY)

net = Net()

cov_1 = net.CreateConvolutionLayer(
    input_data=trainingSet.X[0], coreSize=(3, 3), coreNumber=16
)
Y1 = cov_1._Convolution()

pool_1 = net.CreatePoolingLayer(
    input_data=Y1,
    coreSize=(2, 2),
)
Y2 = pool_1._pooling()

cov_2 = net.CreateConvolutionLayer(input_data=Y2, coreSize=(3, 3), coreNumber=32)
Y3 = cov_2._Convolution()

pool_2 = net.CreatePoolingLayer(
    input_data=Y3,
    coreSize=(2, 2),
)
Y4 = pool_2._pooling()
Y5 = Y4.flatten()
# Y5 = pool_2._pooling()
full_1 = net.CreateFullyConnectedLayer(Y5, 10)
full_1.Dense()

# flateen
# Dropout
aaa = 1
