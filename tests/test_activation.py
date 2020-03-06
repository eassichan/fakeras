# -*- coding: utf-8 -*-

import os
import sys
from unittest import TestCase

import numpy as np
import pytest as pt

from fakeras.activation import ReLU, Sigmoid, Softmax
from fakeras.exception import BackwardBeforeForward, TensorShapeNotMatch

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


class TestReLU(TestCase):
    """ 测试fakeras.activation.ReLU. """

    def setUp(self):
        self.relu = ReLU()

    def test_backward(self):
        ug = np.array([[1.0, 66.0], [3.0, 13.0]])
        x = np.array([[4.0, -12.0], [-1.0, 8.0]])
        t = np.array([[1.0, 0.0], [0.0, 13.0]])
        self.relu.forward(x)
        grad = self.relu.backward(ug)
        assert (grad == t).all()

    def test_backward_raise_typeerror(self):
        ug = 1
        x = np.array([[4.0, -12.0], [-1.0, 8.0]])
        self.relu.forward(x)
        with pt.raises(TypeError):
            self.relu.backward(ug)

    def test_backward_raise_backwardbeforeforward(self):
        assert self.relu.inputs is None
        ug = np.array([[1.0, 66.0], [3.0, 13.0]])
        with pt.raises(BackwardBeforeForward) as e:
            self.relu.backward(ug)
        err_msg = "Please execute forward propagation before backward propagation."
        assert e.value.args[0] == err_msg

    def test_backward_raise_tensorshapenotmatch(self):
        ug = np.array([[1.0, 66.0]])
        x = np.array([[4.0, -12.0], [-1.0, 8.0]])
        self.relu.forward(x)
        with pt.raises(TensorShapeNotMatch) as e:
            self.relu.backward(ug)
        assert (
            e.value.args[0]
            == "The shape of target tensor(1, 2) is not match reference tensor(2, 2)."
        )

    def test_forward(self):
        x = np.array([[4.0, -12.0], [-1.0, 8.0]])
        y = self.relu(x)
        t = np.array([[4.0, 0.0], [0.0, 8.0]])
        assert (y == t).all()

    def test_forward_raise_typeerror(self):
        x = 1.0
        with pt.raises(TypeError):
            y = self.relu(x)


class TestSigmoid(TestCase):
    """ 测试fakeras.activation.sigmoid. """

    def setUp(self):
        self.sigmoid = Sigmoid()

    def test_backward(self):
        x = np.array([[4.0, -12.0], [-1.0, 8.0]])
        y = self.sigmoid(x)
        ug = np.array([[1.0, 1.0], [1.0, 1.0]])
        grad = self.sigmoid.backward(ug)
        assert (grad == y * (1 - y)).all()

    def test_backward_raise_backwardbeforeforward(self):
        assert self.sigmoid.outputs is None
        ug = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pt.raises(BackwardBeforeForward):
            self.sigmoid.backward(ug)

    def test_backward_raise_tensorshapenotmatch(self):
        x = np.array([[4.0, -12.0], [-1.0, 8.0]])
        self.sigmoid(x)
        ug = np.array([[1.0, 1.0]])
        with pt.raises(TensorShapeNotMatch):
            self.sigmoid.backward(ug)

    def test_backward_raise_typeerror(self):
        x = 1.0
        with pt.raises(TypeError):
            self.sigmoid(x)

    def test_forward(self):
        x = np.array([[4.0, -12.0], [-1.0, 8.0]])
        y = self.sigmoid(x)
        t = np.array(
            [[9.82013790e-01, 6.14417460e-06], [2.68941421e-01, 9.99664650e-01]]
        )
        assert str(y) == str(t)

    def test_forward_raise_typeerror(self):
        x = None
        with pt.raises(TypeError):
            self.sigmoid(x)


class TestSoftmax(TestCase):
    """ 测试fakeras.activation.softmax. """

    def setUp(self):
        self.softmax = Softmax()

    def test_backward(self):
        x = np.array([[4.0, -2.0, 3.0], [-1.0, 2.4, 1.2]])
        y = self.softmax(x)
        t = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ug = -t / (y * y.shape[0])
        grad = self.softmax.backward(ug)
        assert str(grad) == str((y - t) / y.shape[0])

    def test_backward_raise_backwardbeforeforeward(self):
        assert self.softmax.outputs is None
        ug = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        assert self.softmax.outputs is None
        with pt.raises(BackwardBeforeForward):
            self.softmax.backward(ug)

    def test_backward_raise_tensorshapenotmatch(self):
        x = np.array([[4.0, -2.0, 3.0], [-1.0, 2.4, 1.2]])
        self.softmax(x)
        ug = np.random.random((3, 3))
        with pt.raises(TensorShapeNotMatch):
            self.softmax.backward(ug)

    def test_backward_raise_typeerror(self):
        x = np.array([[4.0, -2.0, 3.0], [-1.0, 2.4, 1.2]])
        self.softmax(x)
        ug = 404
        with pt.raises(TypeError):
            self.softmax.backward(ug)

    def test_forward(self):
        x = np.array([[4.0, -12.0], [-1.0, 8.0]])
        y = self.softmax(x)
        t = np.array(
            [[9.99999887e-01, 1.12535162e-07], [1.23394576e-04, 9.99876605e-01]]
        )
        assert str(y) == str(t)

    def test_forward_raise_typeerror(self):
        x = False
        with pt.raises(TypeError):
            self.softmax(x)

    def test_forward_raise_valueerror(self):
        x = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
        with pt.raises(ValueError):
            self.softmax(x)
