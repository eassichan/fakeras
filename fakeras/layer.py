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


class Conv2D(Layer):
    """ 二维卷积层.

    Parameters
    ----------
    fitler_num:int > 0
        过滤器/卷积核总数.
    filter_size: tuple
        过滤器/卷积核的高与宽.
    activation: str or subclass of fakeras.activation.Activation
        激活函数.
    initialization: str
        权重初始化方法.
    input_shape: tuple
        输入张量的形状，二维卷积层输入张量的维度为（通道数，高度，宽度）.
    pad: int
        滑动窗口/局部感受野的填充宽度.
    stride: int
        活动窗口/局部感受野的滑动步长.

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
    batch_size: int
        批量样本数.
    biases: numpy.ndarray
        当前层的偏置.
    built: boolean
        当前层是否已经创建的状态标志位.
    fn, fh, fw: int, int, int
        过滤器/卷积核的总数，高度和宽度.
    grads: dict
        当前层的梯度，包括损失函数对输入的梯度，损失函数对权重的梯度，以及损失函数对偏置的梯度.
    ic, ih, iw: int, int, int
        出入特征图的通道数，高度和宽度.
    initialize: callable
        权重参数初始化方法，这些方法在fakeras.initialization模块实现.
    input_expanded: numpy.ndarray
        将四维输入特征图（样本数，输入通道数，输入高度，输入宽度）的局部感受野/滑动窗口展开成按
        行排列的矩阵（样本数 * 输出高度 * 输出宽度，输入通道数 * 卷积核高度 * 卷积核宽度）.
    input_shape: tuple
        输入张量的形状，用于确定权重参数张量的形状.
    oh, ow: int, int
        输出特征图的高度和输出宽度.
    output_shape: tuple
        输出张量的形状，主要用于构建神经网络时连接不同的层.
    pad: int
        局部感受野/滑动窗口的填充宽度.
    stride: int
        滑动窗口的滑动步长.
    weights: numpy.ndarray
        当前层的权重参数.
    weights_reshaped: numpy.ndarray
        将权重参数张量重塑成可与input_expanded做点积运算的形状.

    Methods
    -------
    backward:
        反向传播.
    build:
        根据built状态位判断是否构建当前层.
    _build:
        根据输入形状计算输出形状，并初始化权重等参数，最后将状态设置为已创建.
    forward:
        正向传播.
    """

    def __init__(
        self,
        filter_num,
        filter_size,
        activation,
        initialization,
        input_shape=None,
        pad=0,
        stride=1,
    ):
        self.weights = None
        self.biases = None
        self.grads = None

        self.oh = None
        self.ow = None
        self.input_shape = None
        self.output_shape = None

        self.batch_size = None
        self.input_expanded = None
        self.weight_reshaped = None

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

        validate_data_type(tuple, 'filter_size', filter_size)

        self.fn = filter_num
        self.fh, self.fw = filter_size
        self.pad = pad
        self.stride = stride

        if input_shape is not None:
            validate_data_type(tuple, 'input_shape', input_shape)
            self._build(input_shape=input_shape)
        else:
            self.built = False

    def _build(self, input_shape):
        validate_data_type(tuple, "input_shape", input_shape)
        if len(input_shape) != 3:
            err_msg = "Input feature map must be a 3D tensor, but %sD is found." % len(
                input_shape
            )
            raise ValueError(err_msg)

        self.ic, self.ih, self.iw = input_shape
        self.oh = (self.ih + 2 * self.pad - self.fh) // self.stride + 1
        self.ow = (self.iw + 2 * self.pad - self.fw) // self.stride + 1
        self.input_shape = input_shape
        self.output_shape = (self.fn, self.oh, self.ow)

        self.weights = self.initialize((self.fn, self.ic, self.fh, self.fw))
        self.biases = np.zeros(self.fn)

        self.built = True

    def backward(self, upstream_gradient, weight_decay_lambda=0.0):
        validate_data_type(np.ndarray, "upstream_gradient", upstream_gradient)
        self._validate_propagation_order(
            forward_done=(
                self.input_expanded is not None and self.weight_reshaped is not None
            )
        )

        # upstream_gradient的形状：(bn, fn, oh, ow)
        ug = upstream_gradient.transpose(0, 2, 3, 1)  # (bn, oh, ow, fn)
        ug = ug.reshape(self.batch_size * self.oh * self.ow, self.fn)

        # weigth_reshaped的形状：(ic * fh * fw, fn)
        ig = np.dot(ug, self.weight_reshaped.T)  # (bn * oh * ow, ic * fh * fw)
        input_shape = (self.batch_size, self.ic, self.ih, self.iw)
        ig = row2im(
            row=ig,
            input_shape=input_shape,
            fh=self.fh,
            fw=self.fw,
            pad=self.pad,
            stride=self.stride
        )

        # input_expanded的形状：(bn * oh * ow, ic * ih * iw)
        wg = np.dot(self.input_expanded.T, ug)  # (ic * ih * iw, fn)
        wg = wg.T.reshape(self.fn, self.ic, self.fh, self.fw)
        wg = wg + weight_decay_lambda * self.weights

        bg = ug.sum(axis=0)
        grads = dict(i=ig, w=wg, b=bg)
        self.grads = grads
        return grads

    def forward(self, inputs):
        validate_data_type(np.ndarray, "inputs", inputs)

        self.batch_size = inputs.shape[0]
        self.input_expanded = im2row(
            inputs=inputs,
            fh=self.fh,
            fw=self.fw,
            pad=self.pad,
            stride=self.stride
        )  # (bn * oh * ow, ic * ih * iw)
        self.weight_reshaped = self.weights.reshape(
            self.fn, self.ic * self.fh * self.fw
        ).T  # (ic * fh * fw, fn)
        outputs = np.dot(self.input_expanded, self.weight_reshaped) + self.biases
        outputs = outputs.reshape(self.batch_size, self.oh, self.ow, self.fn)
        outputs = outputs.transpose(0, 3, 1, 2)  # (bn, fn, oh, ow)
        return outputs


class MaxPool2D(Layer):
    """ 二维池化层.

    Parameters
    ----------
    pool_size: tuple
        池化窗口的高与宽.
    input_shape: tuple
        输入张量的形状，二维池化层输入张量的维度为（通道数，高度，宽度）.
    stride: int
        活动窗口/局部感受野的滑动步长.

    Attributes
    ----------
    batch_size: int
        批量样本数.
    built: boolean
        当前层是否已经创建的状态标志位.
    grads: numpy.ndarray
        当前层的梯度，二维池化层只包含损失函数对输入的梯度.
    ic, ih, iw: int, int, int
        出入特征图的通道数，高度和宽度.
    input_shape: tuple
        输入张量的形状，用于确定权重参数张量的形状.
    max_indices: numpy.ndarray
        保存每个局部感受野的最大值的列索引.
    oh, ow: int, int
        输出特征图的高度和输出宽度.
    output_shape: tuple
        输出张量的形状，主要用于构建神经网络时连接不同的层.
    ph, pw: int, int
        池化窗口的高度和输出宽度.
    pad: int
        局部感受野/滑动窗口的填充宽度.
    stride: int
        滑动窗口的滑动步长.
    weights: numpy.ndarray
        当前层的权重参数.

    Methods
    -------
    backward:
        反向传播.
    build:
        根据built状态位判断是否构建当前层.
    _build:
        根据输入形状计算输出形状，并初始化权重等参数，最后将状态设置为已创建.
    forward:
        正向传播.
    """

    def __init__(self, pool_size, input_shape=None, stride=None):
        validate_data_type(tuple, 'pool_size', pool_size)

        self.ph, self.pw = pool_size
        self.input_shape = None
        self.output_shape = None
        self.ic = self.ih = self.iw = None

        self.batch_size = None
        self.max_indices = None
        self.grads = None

        if stride is None:
            self.stride = self.ph
        else:
            self.stride = stride

        if input_shape is not None:
            validate_data_type(tuple, 'input_shape', input_shape)
            self._build(input_shape=input_shape)
        else:
            self.built = False

    def _build(self, input_shape):
        validate_data_type(tuple, "input_shape", input_shape)
        if len(input_shape) != 3:
            err_msg = "Input feature map must be a 3D tensor, but %sD is found." % len(
                input_shape
            )
            raise ValueError(err_msg)

        self.ic, self.ih, self.iw = input_shape
        self.oh = (self.ih - self.ph) // self.stride + 1
        self.ow = (self.iw - self.pw) // self.stride + 1
        self.input_shape = input_shape
        self.output_shape = (self.ic, self.oh, self.ow)

        self.built = True

    def backward(self, upstream_gradient, weight_decay_lambda=0.0):
        # 参考《深度学习入门——基于Python的理论与实现》P223图7-22的逆过程
        validate_data_type(np.ndarray, "upstream_gradient", upstream_gradient)

        grad = np.zeros(
            (self.batch_size * self.ic * self.oh * self.ow, self.ph * self.pw)
        )
        ug = upstream_gradient.flatten()  # 将张量降维成1维
        grad[np.arange(self.max_indices.size), self.max_indices] = ug
        grad = grad.reshape(
            self.batch_size, self.ic, self.oh, self.ow, self.ph, self.pw
        )
        grad = grad.transpose(0, 2, 3, 1, 4, 5)  # (bn, oh, ow, ic, ph, pw)
        grad = grad.reshape(
            self.batch_size * self.oh * self.ow, self.ic * self.ph * self.pw
        )
        input_shape = (self.batch_size, self.ic, self.ih, self.iw)
        grad = row2im(
            row=grad,
            input_shape=input_shape,
            fh=self.ph,
            fw=self.pw,
            stride=self.stride
        )
        grads = dict(i=grad)
        self.grads = grads
        return grads

    def forward(self, inputs):
        # 参考《深度学习入门——基于Python的理论与实现》P223的图7-22
        validate_data_type(np.ndarray, "inputs", inputs)

        self.batch_size = inputs.shape[0]
        input_expanded = im2row(
            inputs=inputs,
            fh=self.ph,
            fw=self.pw,
            stride=self.stride
        )  # (bn * oh * ow, ic * ph * pw)
        input_expanded = input_expanded.reshape(
            self.batch_size, self.oh, self.ow, self.ic, self.ph, self.pw
        )
        input_expanded = input_expanded.transpose(0, 3, 1, 2, 4, 5) # (bn, ic, oh, ow, ph, pw)
        input_expanded = input_expanded.reshape(-1, self.ph * self.pw)

        # 将每个池化窗口展开成行，然后取每一行的最大值：
        self.max_indices = input_expanded.argmax(axis=1)  # 1D
        outputs = input_expanded.max(axis=1)
        outputs = outputs.reshape(self.batch_size, self.ic, self.oh, self.ow)
        return outputs


class Flatten(Layer):
    """ 扁平化层.

    Parameters
    ----------
    input_shape: tuple
        输入张量的形状，默认值为None.

    Attributes
    ----------
    batch_size: int
        批量样本数.
    built: boolean
        当前层是否已经创建的状态标志位.
    forward_done: boolean
        反向传播是否已经完成的状态标志位.
    grads: numpy.ndarray
        当前层的梯度，扁平化层只包含损失函数对输入的梯度.
    input_shape: tuple
        输入张量的形状.
    output_shape: tuple
        输出张量的形状.

    Methods
    -------
    backward:
        反向传播.
    build:
        根据built状态位判断是否构建当前层.
    _build:
        根据输入形状计算输出形状，并初始化权重等参数，最后将状态设置为已创建.
    forward:
        正向传播.
    """

    def __init__(self, input_shape=None):
        self.batch_size = None
        self.output_shape = None

        self.forward_done = False
        self.grads = None

        if input_shape is None:
            self.built = False
        else:
            validate_data_type(tuple, 'input_shape', input_shape)
            self._build(input_shape)

    def _build(self, input_shape):
        validate_data_type(tuple, "input_shape", input_shape)

        self.input_shape = input_shape
        self.output_shape = (np.prod(input_shape),)

        self.built = True

    def backward(self, upstream_gradient, weight_decay_lambda=0.0):
        validate_data_type(np.ndarray, "upstream_gradient", upstream_gradient)
        self._validate_propagation_order(forward_done=self.forward_done)

        input_shape = (self.batch_size,) + self.input_shape
        ig = upstream_gradient.reshape(*input_shape)
        grads = dict(i=ig)
        self.grads = grads
        self.forward_done = False
        return grads

    def forward(self, inputs):
        validate_data_type(np.ndarray, "inputs", inputs)

        self.batch_size = inputs.shape[0]
        output_shape = (self.batch_size,) + self.output_shape
        outputs = inputs.reshape(*output_shape)
        self.forward_done = True
        return outputs


def im2row(inputs, fh, fw, pad=0, stride=1):
    """ 将输入特征图的局部感受野展开成行.

    Parameters
    ----------
    inputs: numpy.ndarray
        输入张量，形状为（样本数，输入通道数，输入高度，输入宽度）.
    fh: int
        filter height，过滤器/卷积核的高度.
    fw: int
        filter weight，过滤器/卷积核的宽度.
    pad: int
        过滤器/卷积核的填充宽度，默认为0.
    stride: int
        滑动窗口的滑动步长，默认值为1.

    Returns
    -------
    row: numpy.ndarray
        将四维输入特征图（样本数，输入通道数，输入高度，输入宽度）的局部感受野/滑动窗口展开成按
        行排列的矩阵（样本数 * 输出高度 * 输出宽度，输入通道数 * 卷积核高度 * 卷积核宽度）.

    Raises
    ------
    ValueError:
        输入参数中的过滤器/卷积核的宽和高为负数时抛出.
    """
    validate_data_type(np.ndarray, "inputs", inputs)
    if fh <= 0 or fw <= 0:
        err_msg = (
            "The height and width of the convolution kernel must be greater than 0."
        )
        raise ValueError(err_msg)

    bn, ic, ih, iw = inputs.shape  # batch num, input channel, input height, input width
    oh = (ih + 2 * pad - fh) // stride + 1
    ow = (iw + 2 * pad - fw) // stride + 1

    img = np.pad(inputs, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    row = np.zeros((bn, oh, ow, ic, fh, fw))
    for i in range(oh):  # oh决定了卷积核在输入特征图的垂直方向上可以移动多少次
        row_start = i * stride
        row_stop = row_start + fh
        for j in range(ow):  # ow决定了卷积核在输入特征图的水平方向上可以移动多少次
            col_start = j * stride
            col_stop = col_start + fw
            receptive_field = img[:, :, row_start:row_stop, col_start:col_stop]
            row[:, i, j, :, :, :] = receptive_field
    row = row.reshape(bn * oh * ow, -1)
    return row


def row2im(row, input_shape, fh, fw, pad=0, stride=1):
    """ 将已经展开成行的局部感受野还原为输入特征图.

    Parameters
    ----------
    row: numpy.ndarray
        将四维输入特征图（样本数，输入通道数，输入高度，输入宽度）的局部感受野/滑动窗口展开成按
        行排列的矩阵（样本数 * 输出高度 * 输出宽度，输入通道数 * 卷积核高度 * 卷积核宽度）.
    input_shape: tuple
        输入张量的形状，四维张量（样本数，输入通道数，输入高度，输入宽度）.
    fh: int
        过滤器/卷积核的高度.
    fw: int
        过滤器/卷积核的宽度.
    pad: int
        过滤器/卷积核的填充宽度，默认为0.
    stride: int
        滑动窗口的滑动步长，默认值为1.

    Returns
    -------
    img: numpy.ndarray
        将展开矩阵还原为原始格式后的特征图.

    Raises
    ------
    ValueError:
        以下情况都会抛出该异常：
        1. 输入参数中的过滤器/卷积核的宽和高为负数时抛出；
        2. 展开式不是一个矩阵.
    """
    validate_data_type(np.ndarray, "row", row)
    if len(row.shape) != 2:
        err_msg = (
            "Expansion of input feature map must be a 2D tensor, \
                   but %sD tensor is found."
            % row.ndim
        )
        raise ValueError(err_msg)
    validate_data_type(tuple, "input_shape", input_shape)
    if fh <= 0 or fw <= 0:
        err_msg = (
            "The height and width of the convolution kernel must be greater than 0."
        )
        raise ValueError(err_msg)

    bn, ic, ih, iw = input_shape  # batch num, input channel, input height, input width
    oh = (ih + 2 * pad - fh) // stride + 1
    ow = (iw + 2 * pad - fw) // stride + 1

    row = row.reshape(bn, oh, ow, ic, fh, fw)
    img = np.zeros((bn, ic, ih + 2 * pad, iw + 2 * pad))
    for i in range(oh):
        row_start = i * stride
        row_stop = row_start + fh
        for j in range(ow):
            col_start = j * stride
            col_stop = col_start + fw
            receptive_field = row[:, i, j, :, :, :]
            img[:, :, row_start:row_stop, col_start:col_stop] = receptive_field
    img = img[:, :, pad:pad+ih, pad:pad+iw]
    return img
