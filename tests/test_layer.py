# -*- coding: utf-8 -*-

import os
import sys
from unittest import TestCase

import numpy as np
import pytest as pt

from fakeras.activation import ACTIVATIONS as ACTI
from fakeras.exception import (
    ActivationFunctionNotFound,
    BackwardBeforeForward,
    EmptyInputShape,
    InitializationMethodNotFound,
)
from fakeras.initialization import INITIALIZATIONS as INIT
from fakeras.layer import Dense

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


class TestDense(TestCase):
    """ Unit test for fakeras.layer.Dense. """

    def setUp(self):
        self.dense = Dense(512, activation="relu", initialization="he")

    def test_init_raise_activationfunctionnotfound(self):
        with pt.raises(ActivationFunctionNotFound):
            assert "elu" not in ACTI
            Dense(512, activation="elu", initialization="he")

    def test_init_raise_initializationmethodnotfound(self):
        with pt.raises(InitializationMethodNotFound):
            assert "he_uniform" not in INIT
            Dense(512, activation="relu", initialization="he_uniform")

    def test__build(self):
        # input_shape = (1000, 784)
        input_shape = (784,)
        self.dense._build(input_shape)
        assert self.dense.built is True
        assert self.dense.weights.shape == (784, 512)
        assert self.dense.biases.shape == (1, 512)
        assert self.dense.output_shape == (512,)

    def test__build_raise_valueerror(self):
        input_shape = (1000, 28, 28)
        with pt.raises(ValueError):
            self.dense._build(input_shape)

    # def test_build_with_batchsize(self):
    #     dense = Dense(64, 'relu', 'he', input_shape=(784,))
    #     assert dense.built is False
    #     batch_size = 60000
    #     dense.build(batch_size=batch_size)
    #     assert dense.built is True
    #     assert dense.input_shape == (60000, 784)
    #     assert dense.output_shape == (60000, 64)
    #     assert dense.weights.shape == (784, 64)
    #     assert dense.biases.shape == (1, 64)

    # def test_build_with_batchsize_raise_emptyinputshape(self):
    #     assert self.dense.built is False
    #     batch_size = 60000
    #     with pt.raises(EmptyInputShape):
    #         self.dense.build(batch_size=batch_size)

    def test_build(self):
        assert self.dense.built is False
        input_shape = (784,)
        self.dense.build(input_shape=input_shape)
        assert self.dense.built is True
        assert self.dense.output_shape == (512,)
        assert self.dense.weights.shape == (784, 512)
        assert self.dense.biases.shape == (1, 512)

    def test_build_raise_emptyinputshape(self):
        with pt.raises(EmptyInputShape):
            input_shape = None
            self.dense.build(input_shape)

    def test_forward(self):
        # x = np.random.random((784, 600))
        input_shape = (784,)
        self.dense.build(input_shape=input_shape)

        x = np.random.random((1000, 784))
        y = self.dense.forward(x)
        t = np.dot(x, self.dense.weights) + self.dense.biases
        t[t <= 0] = 0  # ReLU
        assert (y == t).all()

    def test_backward(self):
        dense = Dense(2, activation="identity", initialization="he", input_shape=(2,))
        dense.weights = np.array([[5, 6], [7, 8]])  # 使用可逆的权重矩阵
        x = np.array([[1, 2], [3, 4]])  # x也必须可逆且全部元素都大于0
        y = dense.forward(x)
        ug = np.random.random(y.shape)
        grads = dense.backward(ug)
        ig, wg, bg = grads["i"], grads["w"], grads["b"]
        assert str(ug) == str(np.dot(np.linalg.inv(x.T), wg))
        assert str(ug) == str(np.dot(ig, np.linalg.inv(dense.weights.T)))
