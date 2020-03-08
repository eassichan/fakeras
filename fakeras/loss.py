# -*- coding: utf-8 -*-

"""
fakeras.loss
~~~~~~~~~~~

本模块实现了神经网络的训练中常用的损失函数.
"""

import numpy as np

from .abstract import ComputeNode
from .exception import compare_tensor_shape, validate_data_type


class Loss(ComputeNode):
    """ 损失函数基类. """

    pass


class BinaryCrossEntropy(Loss):
    """ 二元分类交叉熵损失函数.

    Attributes
    ----------
    delta: float
        防止被零除而在分母中添加的一个微小值.
    y: numpy.ndarray
        神经网络的预测值，用于反向传播的梯度计算.
    t: numpy.ndarray
        神经网络的目标值，用于反向传播的梯度计算.

    Methods
    -------
    __call__:
        调用forward计算损失误差.
    backward:
        反向传播，求损失函数对预测值的梯度.
    forward:
        正向传播，求损失误差.
    """

    def __init__(self):
        self._delta = 1e-9

        self.y = None
        self.t = None

    def __call__(self, y, t):
        e = self.forward(y, t)
        return e

    def backward(self):
        self._validate_propagation_order(
            forward_done=(self.y is not None and self.t is not None)
        )

        # dl/dy = (1-t)/(1-y)-t/y
        grad = (1 - self.t) / (1 - self.y + self._delta) \
               - self.t / (self.y + self._delta)
        grad /= self.y.shape[0]
        return grad

    def forward(self, y, t):
        validate_data_type(np.ndarray, "y", y)
        validate_data_type(np.ndarray, "t", t)
        compare_tensor_shape(reference=y, target=t)  # 张量y和张量t的形状必须相同

        self.y = y
        self.t = t
        e = - np.sum(t * np.log(y + self._delta) + (1 - t) * np.log(1 - y + self._delta))
        e /= y.shape[0]
        return e


class CategoricalCrossEntropy(Loss):
    """ 分类交叉熵损失函数.

    分类交叉熵的公式定义：
        e = - (t11*ln(y11) + t12*ln(y12) + ... + t21*ln(y21) + ... + tij*ln(yij)
               + ... + tmn*ln(ymn)) / m
        其中，m和n分别为样本总数和分类类别总数.

    Attributes
    ----------
    delta: float
        防止被零除而在分母中添加的一个微小值.
    y: numpy.ndarray
        神经网络的预测值，用于反向传播的梯度计算.
    t: numpy.ndarray
        神经网络的目标值，用于反向传播的梯度计算.

    Methods
    -------
    __call__:
        调用forward计算损失误差.
    backward:
        反向传播，求损失函数对预测值的梯度.
    forward:
        正向传播，求损失误差.
    """

    def __init__(self):
        self._delta = 1e-9

        self.y = None
        self.t = None

    def __call__(self, y, t):
        e = self.forward(y, t)
        return e

    def backward(self):
        self._validate_propagation_order(
            forward_done=(self.y is not None and self.t is not None)
        )

        grad = - self.t / (self.y + self._delta)
        grad /= self.y.shape[0]
        return grad

    def forward(self, y, t):
        validate_data_type(np.ndarray, "y", y)
        validate_data_type(np.ndarray, "t", t)
        compare_tensor_shape(reference=y, target=t)

        self.y = y
        self.t = t
        e = - np.sum(t * np.log(y + self._delta))  # loss error
        e /= y.shape[0]
        return e


class MeanSquareError(Loss):
    """ 均方误差损失函数.

    均方误差损失函数公式定义：
        e = ((y1-t1)^2 + (y2-t2)^2 + ... + (yi-ti)^2 + ... + (ym-tm)^2) / 2*m
        其中m为样本总数.

    Attributes
    ----------
    y: numpy.ndarray
        神经网络的预测值，用于反向传播的梯度计算.
    t: numpy.ndarray
        神经网络的目标值，用于反向传播的梯度计算.

    Methods
    -------
    __call__:
        调用forward计算损失误差.
    backward:
        反向传播，求损失函数对预测值的梯度.
    forward:
        正向传播，求损失误差.
    """

    def __init__(self):
        self.y = None
        self.t = None

    def __call__(self, y, t):
        e = self.forward(y, t)
        return e

    def backward(self):
        self._validate_propagation_order(
            forward_done=(self.y is not None and self.t is not None)
        )

        grad = (self.y - self.t) / self.y.shape[0]
        return grad

    def forward(self, y, t):
        validate_data_type(np.ndarray, "y", y)
        validate_data_type(np.ndarray, "t", t)
        compare_tensor_shape(y, t)  # y and t should both have the same shape

        self.y = y
        self.t = t
        # e = 0.5 * np.sum(np.square(y - t)) / y.shape[0]
        e = np.sum(np.square(y - t))
        e /= (2 * y.shape[0])
        return e


LOSSES = {
    "bce": BinaryCrossEntropy,
    "binarycrossentropy": BinaryCrossEntropy,
    "cce": CategoricalCrossEntropy,
    "categoricalcrossentropy": CategoricalCrossEntropy,
    "mse": MeanSquareError,
    "meansquareerror": MeanSquareError,
}
