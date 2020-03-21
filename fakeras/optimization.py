# -*- coding: utf-8 -*-

"""
fakeras.optimization
~~~~~~~~~~~~~~~~~~~~

本模块实现了一些经典常用的神经网络优化方法.
"""

import abc

import numpy as np

from .exception import compare_tensor_shape


class Optimizer:
    """ 神经网络优化器基类，更新各层的参数. """

    @abc.abstractmethod
    def collect_update_variables(self):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, param=None, grad=None):
        raise NotImplementedError


class GD(Optimizer):
    """ 梯度下降优化器.

    这里的梯度下降只提供最基本的参数更新功能，参数的更新功能对于批量梯度下降（BGD），
    随机梯度下降（SGD）以及小批量梯度下降（Mini-batch GD）都是相同的，不同的只是对
    训练集的划分. 在fakeras，对训练集的随机打乱、分批等功能并不包含在优化器当中，而是
    集成在与网络的训练相关的fakeras.network模块.

    Parameters
    ----------
    network: fakeras.network.NeuralNetwork
        需要优化的神经网络.
    lr: float
        learning rate，梯度下降算法中的学习率.

    Attributes
    ----------
    grads: list of dict
        保存神经网络各层梯度的列表，梯度包括损失函数对权重参数和偏置参数的梯度.
    lr: float
        梯度下降算法中的学习率，默认值与keras的SGD算法一样，设为0.01.
    network: fakeras.network.NeuralNetwork
        需要优化的神经网络.
    params: list of dict
        保存神经网络各层可训练参数（trainable）的列表，在fakeras，这些参数包括权重和偏置.

    Methods
    -------
    collect_update_variables:
        收集神经网络各层可训练参数及其对应的梯度.
    update:
        更新神经网络的参数.
    """

    def __init__(self, network, lr=0.01):
        self.lr = lr
        self.network = network

    def collect_update_variables(self):
        self.params = []
        self.grads = []
        for layer in self.network:
            if hasattr(layer, "weights") and \
                hasattr(layer, "biases") and \
                hasattr(layer, "grads"):
                param = dict(w=layer.weights, b=layer.biases)
                self.params.append(param)
                self.grads.append(layer.grads)
            # else:
            #     self.params.append(None)
            #     self.grads.append(None)

    def update(self, param=None, grad=None):
        self.collect_update_variables()
        for i in range(len(self.network)):
            p = self.params[i]
            g = self.grads[i]
            if p is not None and g is not None:
                for k in p:
                    p[k] -= self.lr * g[k]


class AdaGrad(Optimizer):

    """ AdaGrad优化器.

    # TODO
    """

    def collect_update_variables(self):
        pass

    def update(self, param, grad):
        pass


class Adam(Optimizer):
    """ Adam优化器.

    # TODO
    """

    def collect_update_variables(self):
        pass

    def update(self, param, grad):
        pass


class Momentum(Optimizer):
    """ Momentum优化器.

    # TODO
    """

    def collect_update_variables(self):
        pass

    def update(self, param, grad):
        pass


class Nesterov(Optimizer):
    """ Netserov优化器.

    # TODO
    """

    def collect_update_variables(self):
        pass

    def update(self, param, grad):
        pass


class RMSProp(Optimizer):
    """ RMSProp优化器.

    RMSProp算法：
    accumulator := beta * accumulator + (1 - beta) * grad * grad
    param := param - lr * grad / sqrt(accumulator + epsilon)

    Parameters
    ----------
    network: fakeras.network.NeuralNetwork
        需要优化的神经网络.
    beta: float
        优化算法的学习衰减率.
    lr: float
        优化算法的学习率.

    Attributes
    ----------
    accumulators: list of float
        神经网络中各层的RMSProp累积量.
    beta: float
        RMSProp优化算法中的指数衰减率，默认值设为0.9.
    epsilon: float
        防止被零除而在分母上添加的微小值.
    grads: list of dict
        保存神经网络各层梯度的列表，梯度包括损失函数对权重参数和偏置参数的梯度.
    lr: float
        优化算法的学习率，默认值与keras的RMSProp算法一样，设为0.001.
    network: fakeras.network.NeuralNetwork
        需要优化的神经网络.
    params: list of dict
        保存神经网络各层可训练参数（trainable）的列表，这些参数一般包括权重和偏置.

    Methods
    -------
    collect_update_variables:
        收集神经网络各层可训练参数及其对应的梯度.
    update:
        更新神经网络的参数.
    """

    def __init__(self, network, beta=0.9, lr=0.001):
        self.lr = lr
        self.beta = beta
        self.network = network

        self.accumulators = [None for _ in self.network]
        self.epsilon = 1e-9

    def collect_update_variables(self):
        self.params = []
        self.grads = []
        for layer in self.network:
            if (
                hasattr(layer, "weights")
                and hasattr(layer, "biases")
                and hasattr(layer, "grads")
            ):
                param = dict(w=layer.weights, b=layer.biases)
                self.params.append(param)
                self.grads.append(layer.grads)
            else:
                self.params.append(None)
                self.grads.append(None)

    def update(self):
        self.collect_update_variables()
        for i in range(len(self.network)):
            p = self.params[i]
            g = self.grads[i]
            if p is not None and g is not None:
                if self.accumulators[i] is None:
                    accumulator = dict()
                    for k, v in p.items():
                        accumulator[k] = np.zeros_like(v)
                    self.accumulators[i] = accumulator

                for k in p:
                    a = self.accumulators[i][k]
                    a = self.beta * a + (1 - self.beta) * g[k] * g[k]
                    p[k] -= self.lr * g[k] / np.sqrt(a + self.epsilon)
                    self.accumulators[i][k] = a


OPTIMIZERS = {
    "adagrad": AdaGrad,
    "adam": Adam,
    "gd": GD,
    "momentum": Momentum,
    "nesterov": Nesterov,
    "rmsprop": RMSProp,
}
