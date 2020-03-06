# -*- coding: utf-8 -*-

"""
fakeras.activation
~~~~~~~~~~~~~~~~~~

本模块实现了神经网络中一些常用的激活函数和输出函数，包括恒等函数，ReLU，Sigmoid和Softmax.
"""

import abc

import matplotlib.pyplot as plt
import numpy as np

from .abstract import ComputeNode
from .exception import compare_tensor_shape, validate_data_type


class Activation(ComputeNode):
    """ 激活函数抽象基类. """

    @abc.abstractmethod
    def display(self, start, stop, step):
        """ 显示激活函数的函数图像. """
        raise NotImplementedError


class Identity(Activation):
    """ 恒等函数.

    恒等函数定义：y = x，一般多用于标量回归的应用.

    Attributes
    ----------
    inputs: numpy.ndarray
        输入张量，用于反向传播.

    Methods
    -------
    __call__：
        调用Identity实例时直接返回forward的结果.
    backward：
        反向传播求损失函数对输入的梯度.
    display：
        打印函数图像.
    forward：
        正向传播求输出.
    """

    def __init__(self):
        self.inputs = None

    def __call__(self, inputs):
        outputs = self.forward(inputs)
        return outputs

    def backward(self, upstream_gradient):
        validate_data_type(np.ndarray, "upstream_gradient", upstream_gradient)
        self._validate_propagation_order(forward_done=self.inputs)
        compare_tensor_shape(reference=self.inputs, target=upstream_gradient)

        # y=x，dy/dx=1，损失函数对输入的梯度为dl/dx=(dl/dy)*(dy/dx)=dl/dy
        gradient = upstream_gradient
        return gradient

    def display(self, start=-10, stop=11, step=0.01):
        x = np.arange(start, stop, step)
        y = self.forward(x)

        plt.plot(x, y)
        plt.title("Identity function")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    def forward(self, inputs):
        validate_data_type(np.ndarray, "inputs", inputs)

        self.inputs = inputs
        outputs = inputs
        return outputs


class ReLU(Activation):
    """ ReLU（Rectified Linear Unit，修正线性单元）激活函数.

    ReLU函数定义如下：
        y = x, when x > 0
        y = 0, when x <= 0

    Attributes
    ----------
    inputs: numpy.ndarray
        输入张量，用于反向传播时的梯度计算.

    Methods
    -------
    __call__：
        调用ReLU实例时直接返回forward的结果.
    backward：
        反向传播求损失函数对输入的梯度.
    display：
        打印函数图像.
    forward：
        正向传播求输出.
    """

    def __init__(self):
        self.inputs = None

    def __call__(self, inputs):
        outputs = self.forward(inputs)
        return outputs

    def backward(self, upstream_gradient):
        validate_data_type(np.ndarray, "upstream_gradient", upstream_gradient)
        self._validate_propagation_order(forward_done=self.inputs)
        compare_tensor_shape(reference=self.inputs, target=upstream_gradient)

        # 损失函数对输入的梯度为dl/dx=(dl/dy)*(dy/dx)，
        # 当x>0时，有y=x，dy/dx=1，所以dl/dx=dl/dy；
        # 当x<=0时，有y=0，dy/dx=0，所以dl/dx=(dl/dy)*0=0.
        upstream_gradient[self.inputs <= 0] = 0
        return upstream_gradient

    def display(self, start=-10, stop=11, step=0.01):
        x = np.arange(start, stop, step)
        y = self.forward(x)

        plt.plot(x, y)
        plt.title("ReLU function")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    def forward(self, inputs):
        validate_data_type(np.ndarray, "inputs", inputs)

        self.inputs = inputs
        outputs = inputs.copy()
        outputs[inputs <= 0] = 0
        return outputs


class Sigmoid(Activation):
    """ Sigmoid激活函数.

    Sigmoid函数定义：y = 1 / (1 + e^-x)，多用于二元分类.

    Attributes
    ----------
    outputs: numpy.ndarray
        输出张量，用于反向传播时的梯度计算.

    Methods
    -------
    __call__：
        调用Sigmoid实例时直接返回forward的结果.
    backward：
        反向传播求损失函数对输入的梯度.
    display：
        打印函数图像.
    forward：
        正向传播求输出.
    """

    def __init__(self):
        self.outputs = None

    def __call__(self, inputs):
        outputs = self.forward(inputs)
        return outputs

    def backward(self, upstream_gradient):
        validate_data_type(np.ndarray, "upstream_gradient", upstream_gradient)
        self._validate_propagation_order(forward_done=self.outputs)
        compare_tensor_shape(reference=self.outputs, target=upstream_gradient)

        # dl/dx=(dl/dy)*(dy/dx)，而dy/dx=y*(1-y).
        gradient = upstream_gradient * self.outputs * (1 - self.outputs)
        return gradient

    def display(self, start=-10, stop=11, step=0.01):
        x = np.arange(start, stop, step)
        y = self.forward(x)

        plt.plot(x, y)
        plt.title("Sigmoid function")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    def forward(self, inputs):
        if not isinstance(inputs, np.ndarray):
            raise TypeError("Parameter 'inputs' must be an instance of numpy.ndarray.")

        outputs = 1 / (1 + np.exp(-inputs))
        self.outputs = outputs
        return outputs


class Softmax(Activation):
    """ Softmax激活函数.

    softmax函数的定义：
        设x，y均为向量，x=[x1, x2, ..., xn]，y=[y1, y2, ..., yn]，其中：
        yi = e^xi / (e^x1 + e^x2 + ... + e^xn) (1<=i<=n)
        例如，y1 = e^x1 / (e^x1 + e^x2 + ... + e^xn)

    Attributes
    ----------
    outputs: numpy.ndarray
        输出张量，用于反向传播时的梯度计算.

    Methods
    -------
    __call__：
        调用Sofmax实例时直接返回forward的结果.
    backward：
        反向传播求损失函数对输入的梯度.
    display：
        打印函数图像.
    forward：
        正向传播求输出.
    """

    def __init__(self):
        self.outputs = None

    def __call__(self, inputs):
        outputs = self.forward(inputs)
        return outputs

    def backward(self, upstream_gradient):
        validate_data_type(np.ndarray, "upstream_gradient", upstream_gradient)
        self._validate_propagation_order(forward_done=self.outputs)
        compare_tensor_shape(reference=self.outputs, target=upstream_gradient)

        ug = upstream_gradient
        gradient = self.outputs * (ug - np.sum(self.outputs * ug, axis=1, keepdims=True))
        return gradient

    def display(self, start=-5, stop=6, step=0.01):
        x = np.arange(start, stop, step)
        y = self.forward(x)

        plt.plot(x, y)
        plt.title("Softmax activation function")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    def forward(self, inputs):
        validate_data_type(np.ndarray, "inputs", inputs)

        if inputs.ndim == 1:  # 实际中较少用，这里主要用于打印函数图像
            inputs = inputs - np.max(inputs)
            outputs = np.exp(inputs) / np.sum(np.exp(inputs))

        elif inputs.ndim == 2:
            inputs = inputs - np.max(inputs, axis=1, keepdims=True)
            outputs = np.exp(inputs) / np.sum(np.exp(inputs), axis=1, keepdims=True)

        else:
            err_msg = (
                "Parameter 'inputs' must be a 1D or 2D tensor, but %sD tensor is found."
                % inputs.ndim
            )
            raise ValueError(err_msg)

        self.outputs = outputs
        return outputs


ACTIVATIONS = {
    "identity": Identity,
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "softmax": Softmax,
}
