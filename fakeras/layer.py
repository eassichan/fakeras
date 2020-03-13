# -*- coding: utf-8 -*-

"""
fakeras.layer
~~~~~~~~~~~~~

本模块主要实现神经网络中的层，以及构成这些层的公共计算节点.
需要指出的是，这些节点不包含激活函数，有关激活函数的计算节点已经在fakeras.activation模块给出.
"""

import numpy as np

from .abstract import ComputeNode
from .activation import ACTIVATIONS as ACTI
from .exception import (
    ActivationFunctionNotFound,
    EmptyInputShape,
    InitializationMethodNotFound,
    validate_data_type,
)
from .initialization import INITIALIZATIONS as INIT


class Layer(ComputeNode):
    def build(self, input_shape):
        if not self.built:
            if input_shape is not None:
                self._build(input_shape)
            else:
                raise EmptyInputShape


class Dense(Layer):
    """ 神经网络中的全连接层.

    Parameters
    ----------
    neuron_num: int
        可理解为神经网络中当前层的神经元个数，即输出总数.
    activation: str
        当前层选用的激活函数名，这些激活函数由fakeras.activation模块给出.
    initialization: str
        权重参数的初始化方法，这些方法由fakeras.initialization模块给出.
    input_shape: tuple
        输入张量的形状，全连接层输入张量为向量，不包含样本数.

    Raises
    ------
    ActivationFunctionNotFound：
        选用的激活函数未定义或未实现.
    InitializationNotFound:
        选用的初始化方法未定义或未实现.

    Attributes
    ----------
    activator: subclass of fakeras.activation.Activation
        激活函数实例.
    biases: numpy.ndarray
        当前层的偏置.
    built: boolean
        当前层是否已经创建的状态标志位.
    grads: dict
        当前层的梯度，包括损失函数对输入的梯度，损失函数对权重的梯度，以及损失函数对偏置的梯度.
    initialize: callable
        权重参数初始化方法，这些方法在fakeras.initialization模块实现.
    inputs: numpy.ndarray
        每一个训练迭代的输入张量，用于反向传播梯度的计算.
    input_shape: tuple
        输入张量的形状，用于确定权重参数张量的形状.
    neuron_num: int
        神经元个数，也可理解为输出数.
    output_shape: tuple
        输出张量的形状，主要用于构建神经网络时连接不同的层.
    weights: numpy.ndarray
        当前层的权重参数.

    Methods
    -------
    backward:
        反向传播.
    build:
        根据built状态位判断是否构建当前层，主要是确定权重参数的形状以及输出张量的形状.
    _build:
        根据输入形状计算输出形状，并初始化权重等参数，最后将状态设置为已创建.
    forward:
        正向传播.
    """

    def __init__(self, neuron_num, activation, initialization, input_shape=None):
        self.input_shape = None
        self.output_shape = None

        self.weights = None
        self.biases = None

        self.inputs = None
        self.grads = None

        activator = ACTI.get(activation.lower(), None)
        if activator is None:
            err_msg = "Undefined activation function '%s'." % activation
            raise ActivationFunctionNotFound(err_msg)
        self.activator = activator()  # 初始化激活函数实例

        initialize = INIT.get(initialization.lower(), None)
        if initialize is None:
            err_msg = "Undefined initialization method '%s'." % initialization
            raise InitializationMethodNotFound(err_msg)
        self.initialize = initialize

        self.neuron_num = neuron_num
        if input_shape is not None:
            self._build(input_shape=input_shape)
        else:
            self.built = False

    def _build(self, input_shape):
        validate_data_type(tuple, "input_shape", input_shape)
        if len(input_shape) != 1:
            err_msg = "The input tensor for dense layer must be a 1D tensor, "
            "but %sD is found." % len(input_shape)
            raise ValueError(err_msg)

        self.input_shape = input_shape
        self.output_shape = (self.neuron_num,)

        weight_shape = (input_shape[0], self.neuron_num)
        self.weights = self.initialize(weight_shape)
        self.biases = np.zeros((1, self.neuron_num))

        self.built = True

    def backward(self, upstream_gradient, weight_decay_lambda=0.0):
        validate_data_type(np.ndarray, "upstream_gradient", upstream_gradient)
        self._validate_propagation_order(forward_done=(self.inputs is not None))

        upstream_gradient = self.activator.backward(upstream_gradient)
        ig = np.dot(upstream_gradient, self.weights.T)
        wg = (
            np.dot(self.inputs.T, upstream_gradient)
            + weight_decay_lambda * self.weights
        )
        bg = upstream_gradient.sum(axis=0, keepdims=True)
        grads = dict(i=ig, w=wg, b=bg)
        self.grads = grads
        return grads

    def forward(self, inputs):
        validate_data_type(np.ndarray, "inputs", inputs)

        outputs = np.dot(inputs, self.weights) + self.biases
        outputs = self.activator(outputs)
        self.inputs = inputs
        return outputs
