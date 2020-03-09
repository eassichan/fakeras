# -*- coding: utf-8 -*-

import os
import sys
from unittest import TestCase

import numpy as np
import pytest as pt

from fakeras.activation import Sigmoid
from fakeras.metric import BinaryAccuracy, CategoricalAccuracy, MeanAbsoluteError

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))


class TestBinaryAccuracy(TestCase):
    """ fakeras.metric.BinaryAccuracy的单元测试. """

    def setUp(self):
        self.bin_acc = BinaryAccuracy()

    def test__call__(self):
        t = np.random.randn(100, 1)  # 二元分类的输出只有一个，然后假设共有100个样本
        y = t.copy()
        t = np.round(t)
        # 随机改变两个元素的值，使得最终的比例为98%：
        ridx = np.random.choice(y.shape[0], 2)  # row index
        cidx = np.random.choice(y.shape[1], 2)  # column index
        y[ridx, cidx] += 1.0
        acc = self.bin_acc(y, t)
        assert acc == 0.98


class TestCategoricalAccuracy(TestCase):
    """ fakeras.metric.CategoricalAccuracy的单元测试. """

    def setUp(self):
        self.cate_acc = CategoricalAccuracy()

    def test__call__(self):
        y = np.random.randn(100, 10)
        t = np.zeros_like(y)
        max_indices = np.argmax(y, axis=1)
        t[np.arange(len(max_indices)), max_indices] = 1.0  # 构造one-hot编码的标签张量t
        rcs = [(r, c) for r, c in enumerate(max_indices)]
        selected = np.random.choice(len(rcs), 2, replace=False)
        for s in selected:
            rc = rcs[s]
            ridx, cidx = rc[0], rc[1]
            y[ridx, cidx] = 0.0
        acc = self.cate_acc(y, t)
        assert acc == 0.98


class TestMeanAbsoluteError(TestCase):
    """ fakeras.metric.MeanAbsoluteError的单元测试. """

    def setUp(self):
        self.metric = MeanAbsoluteError()

    def test__call__(self):
        t = np.random.randn(100, 1)
        y = t.copy()
        ridx = np.random.choice(t.shape[0], 5, replace=False)
        y[ridx] -= 1.0
        e = self.metric(y, t)
        assert e == 5 / 100
