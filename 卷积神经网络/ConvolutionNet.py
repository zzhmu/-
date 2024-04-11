import numpy as np
import image_load


def ZeroPadding(data, pad):
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


class ConvolutionLayer:
    def __init__(self, size=(3, 3, 1)):
        self.size = size
        self.W = np.random.randn(*size)
        self.b = 0

    def _forward_propagation(self, data, W, Stride=1, Padding=0):
        M, N = data.shape
        m, n = self.W.shape
        I, J = (M - m) / Stride + 1, (N - n) / Stride + 1
        Y = np.zeros((I, J))
        for i in range(I):
            for j in range(J):
                Y[i, j] += self.b
                for u in range(m):
                    for v in range(n):
                        Y[i, j] += (
                            W[u, v]
                            * data[Stride * i + u - Stride, Stride * j + v - Stride]
                        )
        return ReLU(Y)


class PoolingLayer:
    def __init__(self, size=(3, 3, 1)):
        self.size = size

    def _pooling(self, data, Stride=1, Padding=0):
        M, N = data.shape
        m, n = self.W.shape
        I, J = (M - m) / Stride + 1, (N - n) / Stride + 1
        Y = np.zeros((I, J))
        for i in range(I):
            for j in range(J):
                Y[i, j] += self.b
                for u in range(m):
                    for v in range(n):
                        Y[i, j] += data[
                            Stride * i + u - Stride, Stride * j + v - Stride
                        ] / (self.size[0] * self.size[1])
        return Y


class FullyConnectedLayer:
    def __init__(self, data):
        self.data = data
        self.size = data.shape
        self.W = np.random.randn(*self.size)
        self.b = 0

    def _forward_propagation(
        self, W, Stride=1, Padding=0
    ):  # 卷积层和数据层大小一样就是全连接
        M, N = self.size
        m, n = self.W.shape
        I, J = (M - m) / Stride + 1, (N - n) / Stride + 1
        Y = np.zeros((I, J))
        for i in range(M):
            for j in range(N):
                Y[i, j] += self.b
                for u in range(m):
                    for v in range(n):
                        Y[i, j] += (
                            W[u, v]
                            * self.data[
                                Stride * i + u - Stride, Stride * j + v - Stride
                            ]
                        )
        return Y


class ConvolutionNet:
    def __init__(
        self,
        convolutionLayerNumber,
        poolingLayerNumber,
        fullyConnectedLayerNumber,
        trainingSet,
    ):
        self.convolutionLayerNumber = convolutionLayerNumber
        self.poolingLayerNumber = poolingLayerNumber
        self.fullyConnectedLayerNumber = fullyConnectedLayerNumber
        self.trainingSet = trainingSet

    def _createNet(self):
        for i in range(self.convolutionLayerNumber):
            convolutionLayers = ConvolutionLayer()
            convolutionLayers.append(ConvolutionLayer)
        for i in range(self.poolingLayerNumber):
            poolingLayerNumbers = PoolingLayer()
            poolingLayerNumbers.append(PoolingLayer)
        for i in range(self.fullyConnectedLayerNumber):
            fullyConnectedLayers = FullyConnectedLayer()
            fullyConnectedLayers.append(FullyConnectedLayer)


def ReLU(x):
    return np.maximum(0, x)


def Softmax(x):
    exp_values = np.exp(x)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities


input = np.zeros((8, 8, 3))
tensor = np.random.randn(8, 8, 3)
trainingX = [np.random.randn(8, 8, 3) for _ in range(5)]
trainingY = [1, 0, 1, 0, 1]
