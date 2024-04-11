import numpy as np


def ZeroOneLoss(y, f):
    return 1 if y == f else 0


def QuadraticLoss(y, f):
    return 0.5 * (y - f) ** 2


def CrossEntropyLoss(y, f):
    return -sum(y * np.log(f))


def HingeLoss(y, f):
    return max(0, 1 - y * f)
