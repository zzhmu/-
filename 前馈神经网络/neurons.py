# import math

import numpy as np

# class Sigmoid:
# 类的属性

# 类的方法


##################################
def Logistic(x):
    # 函数体，实现函数功能
    return 1 / (1 + np.exp(-x))


def Tanh(x):
    # 函数体，实现函数功能
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def HardLogistic(x):
    # 函数体，实现函数功能
    return max(min(0.25 * x + 0.5, 1), 0)


def HardTanh(x):
    # 函数体，实现函数功能
    return max(min(x, 1), -1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_values = np.exp(
        x - np.max(x, axis=-1, keepdims=True)
    )  # 减去最大值，防止指数溢出
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


######################################


def ReLU(x):
    return np.maximum(0, x)


def LeakyReLU(x, gamma=0.01):
    return max(x, gamma * x)


def PReLU():
    # TODO
    return


def ELU(x, gamma):
    max(0, x) + min(0, gamma * (exp(x) - 1))


def Softplus(x):
    return log(1 + exp(x))


def Swish(x, beta):
    return x * Logistic(beta * x)


def Maxout():
    # TODO
    return
