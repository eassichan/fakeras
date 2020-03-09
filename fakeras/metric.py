# -*- coding: utf-8 -*-

"""
fakeras.metric
~~~~~~~~~~~~~

本模块实现了神经网络训练中用到的一些训练过程监控指标.
"""

import numpy as np

from .exception import compare_tensor_shape, validate_data_type


class Metric:
    """ 训练监控指标基类. """

    pass


class BinaryAccuracy(Metric):
    """ 二元分类识别精度.

    Methods
    -------
    __call__:
        统计预测值和目标值相等的样本个数占总样本个数的比例.
    """

    def __call__(self, y, t):
        validate_data_type(np.ndarray, "y", y)
        validate_data_type(np.ndarray, "t", t)
        compare_tensor_shape(reference=t, target=y)

        acc = np.sum(np.round(y) == t) / y.shape[0]  # t是one-hot编码的，所以需要将y四舍五入
        return acc


class CategoricalAccuracy(Metric):
    """ 多分类识别精度.

    Methods
    -------
    __call__:
        统计预测值和目标值相符的样本个数占总样本个数的比例.
        多分类的预测值是一个向量，包含每个类别的分类概率，而目标值则是一个one-hot编码的向量，
        判断预测值和目标值是否相符只需判断预测向量中的最大值的索引是否与目标向量中元素“1”的
        索引相同.
    """

    def __call__(self, y, t):
        validate_data_type(np.ndarray, "y", y)
        validate_data_type(np.ndarray, "t", t)
        compare_tensor_shape(reference=t, target=y)

        y_max = np.argmax(y, axis=1)
        t_max = np.argmax(t, axis=1)
        acc = np.sum(y_max == t_max) / y.shape[0]
        return acc


class MeanAbsoluteError(Metric):
    """ 平均绝对误差.

    Methods
    -------
    __call__:
        计算预测值和目标值的平均绝对误差.
    """

    def __call__(self, y, t):
        validate_data_type(np.ndarray, "y", y)
        validate_data_type(np.ndarray, "t", t)
        compare_tensor_shape(reference=t, target=y)

        error = np.sum(np.abs(y - t)) / y.shape[0]
        return error


METRICS = {
    "acc": CategoricalAccuracy,
    "accuracy": CategoricalAccuracy,
    "bin_acc": BinaryAccuracy,
    "binary_accuracy": BinaryAccuracy,
    "mae": MeanAbsoluteError,
}
