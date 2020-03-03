# -*- coding: utf-8 -*-

"""
fakeras.abstract
~~~~~~~~~~~~~~~~

本模块给出了fakeras中的一些公共基类或者必须被实现的抽象接口.
"""

import abc

from .exception import BackwardBeforeForward


class ComputeNode:
    """ 神经网络计算图中的抽象计算节点. """

    def _validate_propagation_order(self, forward_done):
        """ 检查正向传播的执行是否先于反向传播.

        反向传播求梯度时，一般需要用到曾参与正向传播的变量或结果，因此在执行流程中，反向传播必须在正向传播
        之后.

        Parameters
        ----------
        forward_done: boolean or None
            正向传播是否已经执行的标记.
        """
        if forward_done is None or forward_done is False:
            err_msg = "Please execute forward propagation before backward propagation."
            raise BackwardBeforeForward(err_msg)

    # @abc.abstractmethod
    def backward(self, upstream_gradient):
        """ 反向传播的运算操作.

        Parameters
        ----------
        upstream_gradient: numpy.ndarray
            张量形式的上游梯度，注意，上游梯度的形状（shape）必须和当前节点的输出张量的形状一致.
        """
        raise NotImplementedError

    # @abc.abstractmethod
    def forward(self, inputs):
        """ 正向传播的运算操作.

        Parameters
        ----------
        inputs: numpy.ndarray
            张量形式的数据，作为当前节点的输入.
        """
        raise NotImplementedError
