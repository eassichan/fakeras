# -*- coding: utf-8 -*-

"""
fakeras.initializer
~~~~~~~~~~~~~~~~~~~

本模块主要实现神经网络训练中经常用到的一些权重参数初始化方法.
"""

import numpy as np


def compute_fans(weight_shape):
    """ 计算训练集的扇入数和扇出数.

    Parameters
    ----------
    weight_shape: tuple
        权重参数张量的形状.

    Returns
    -------
    fan_in: int > 0
        扇入数，即向量数据的特征数或图像数据的输入通道与图像宽高的乘积.
    fan_out: int > 0
        扇出数，即训练向量数据时神经网络层的神经元个数或训练图像数据时卷积核的个数与图像宽高的乘积.

    References
    ----------
    keras.initializers._compute_fans (keras==2.2.4)
    """
    if len(weight_shape) == 2:
        fan_in = weight_shape[0]
        fan_out = weight_shape[1]
    elif len(weight_shape) in {3, 4, 5}:  # 4D张量形状，(output channel, input channel, height, width)
        receptive_field_size = np.prod(weight_shape[2:])  # 局部感受野的大小
        fan_in = weight_shape[1] * receptive_field_size
        fan_out = weight_shape[0] * receptive_field_size
    else:
        fan_in = fan_out = np.sqrt(np.prod(weight_shape))
    return fan_in, fan_out


def he_normalize(weight_shape):
    """ 使用正态分布的何凯明权重参数初始化策略.

    Parameters
    ----------
    weight_shape: tuple
        权重参数张量的形状.

    Returns
    -------
    weights: numpy.ndarray
        初始化后的权重参数张量.
    """
    fan_in, _ = compute_fans(weight_shape)
    std_dev = np.sqrt(2.0 / fan_in)
    weights = np.random.randn(*weight_shape) * std_dev
    return weights


def xavier_uniformize(weight_shape):
    """ 使用均匀分布的Xavier Glorot权重参数初始化策略.

    Parameters
    ----------
    weight_shape: tuple
        权重参数张量的形状.

    Returns
    -------
    weights: numpy.ndarray
        初始化后的权重参数张量.
    """
    fan_in, fan_out = compute_fans(weight_shape)
    high = np.sqrt(6.0 / (fan_in + fan_out))
    low = -high
    weights = np.random.uniform(low, high, weight_shape)
    return weights


def zero_initialize(weight_shape):
    """ 使用全零初始化权重参数.

        深度学习的大多数教程都建议不要使用全零初始化权重参数，这里也只是用来试验.

    Parameters
    ----------
    weight_shape: tuple
        权重参数张量的形状.

    Returns
    -------
    weights: numpy.ndarray
        初始化后的权重参数张量.
    """
    weights = np.zeros(weight_shape)
    return weights


INITIALIZATIONS = {
    "he": he_normalize,
    "xavier": xavier_uniformize,
    "zero": zero_initialize,
}
