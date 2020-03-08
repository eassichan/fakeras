# -*- coding: utf-8 -*-

import os
import sys
from unittest import TestCase

import numpy as np
import pytest as pt

from fakeras.activation import Sigmoid, Softmax
from fakeras.exception import BackwardBeforeForward, TensorShapeNotMatch
from fakeras.loss import BinaryCrossEntropy as BCE
from fakeras.loss import CategoricalCrossEntropy as CCE
from fakeras.loss import MeanSquareError as MSE

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))


class TestBinaryCrossEntropy(TestCase):
    """ Unit test for fakeras.loss.BinaryCrossEntropy. """

    def setUp(self):
        self.bce = BCE()

    def test_backward(self):
        x = np.random.randn(5, 6)
        sigmoid = Sigmoid()
        y = sigmoid(x)
        t = np.random.randn(5, 6)
        self.bce.forward(y, t)
        ug = self.bce.backward()
        dy = sigmoid.backward(ug)
        ref = (y - t) / y.shape[0]
        assert (dy - ref < 1e-7).all()

    def test_backward_raise_backwardbeforeforeward(self):
        assert self.bce.y is None and self.bce.t is None
        with pt.raises(BackwardBeforeForward):
            self.bce.backward()


class TestCategoricalCrossEntropy(TestCase):
    """ Unit test for fakeras.loss.CategoricalCrossEntropy. """

    def setUp(self):
        self.cce = CCE()

    def test_backward(self):
        x = np.random.random((5, 6))
        softmax = Softmax()
        y = softmax(x)
        t = np.zeros_like(y)
        indices = np.argmax(y, axis=1)
        for r, c in enumerate(indices):
            t[r, c] = 1
        self.cce(y, t)
        ug = self.cce.backward()
        dy = softmax.backward(ug)
        # 联合多分类交叉熵损失函数和softmax函数求导的最终结果是：(y - t) / m (m为样本总数)
        ref = (y - t) / y.shape[0]
        assert (dy - ref < 1e-7).all()

    def test_backward_raise_backwardbeforeforeward(self):
        assert self.cce.y is None and self.cce.t is None
        with pt.raises(BackwardBeforeForward):
            self.cce.backward()

    def test_forward(self):
        x = np.random.random((5, 6))
        softmax = Softmax()
        y = softmax(x)
        t = np.zeros_like(y)
        indices = np.argmax(y, axis=1)
        for r, c in enumerate(indices):  # 构造one-hot编码的标签张量t
            t[r, c] = 1
        e = self.cce(y, t)
        e1 = np.round(np.abs(e * y.shape[0]), 4)
        e2 = np.round(np.abs(np.sum(t * np.log(y + self.cce._delta))), 4)
        assert (e1 == e2).all()

    def test_forward_raise_tensorshapenotmatch(self):
        y = np.random.random((5, 6))
        t = np.random.random((5, 4))
        with pt.raises(TensorShapeNotMatch):
            self.cce(y, t)


class TestMeanSquareError(TestCase):
    """ Unit test for fakeras.loss.MeanSquareError. """

    def setUp(self):
        self.mse = MSE()

    def test_backward(self):
        x = np.linspace(-1, 1, 300)
        noise = np.random.randn(*x.shape) * 0.5
        y = 2 * x + 3 + noise
        t = 2 * x + 3
        e = self.mse(y, t)
        grad = self.mse.backward()
        val1 = np.round(grad * y.shape[0], 4)
        val2 = np.round(noise, 4)
        assert (val1 == val2).all()

    def test_backward_raise_backwardbeforeforeward(self):
        assert self.mse.y is None and self.mse.t is None
        with pt.raises(BackwardBeforeForward):
            self.mse.backward()

    def test_forward(self):
        x = np.linspace(-1, 1, 300)  # 300个[-1, 1]之间的浮点数
        noise = np.random.randn(*x.shape) * 0.5  # 随机噪声
        y = 2 * x + 3 + noise
        t = 2 * x + 3
        e = self.mse(y, t)
        e1 = np.round(e * y.shape[0], 4)
        e2 = np.round(0.5 * np.sum(np.square(noise)), 4)
        assert (e1 == e2).all()
