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
from fakeras.layer import Conv2D, Dense, Flatten, MaxPool2D, im2row, row2im

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


class Testim2row(TestCase):

    def setUp(self):
        self.img = np.array([[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]])
        self.inputs = np.zeros((1, 1, 4, 4))
        self.inputs[0, 0, :, :] = self.img
        # print('\n', 'inputs are:')
        # print(self.inputs)
        # print('')

    def test_im2row(self):
        fh = fw = 3
        pad = 0
        stride = 1
        expansion = im2row(inputs=self.inputs, fh=fh, fw=fw, pad=pad, stride=stride)
        target = np.array(
            [
                [1, 2, 3, 0, 1, 2, 3, 0, 1],
                [2, 3, 0, 1, 2, 3, 0, 1, 2],
                [0, 1, 2, 3, 0, 1, 2, 3, 0],
                [1, 2, 3, 0, 1, 2, 3, 0, 1],
            ]
        )
        # print('=' * 80)
        # print(expansion)
        # print('=' * 80)
        assert (expansion == target).all()

    def test_im2row_with_padding(self):
        fh = fw = 3
        pad = 1
        stride = 1
        expansion = im2row(self.inputs, fh, fw, pad, stride)
        # print('\n', 'expansion are:')
        # print(expansion)
        # print('')
        target = np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    3.0,
                    0.0,
                    1.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    3.0,
                    0.0,
                    1.0,
                    2.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    3.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    3.0,
                    0.0,
                    1.0,
                    0.0,
                    2.0,
                    3.0,
                    0.0,
                ],
                [
                    1.0,
                    2.0,
                    3.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    3.0,
                    0.0,
                    1.0,
                    2.0,
                    2.0,
                    3.0,
                    0.0,
                    1.0,
                ],
                [
                    2.0,
                    3.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    3.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    3.0,
                    0.0,
                    1.0,
                    0.0,
                    2.0,
                    3.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    3.0,
                    0.0,
                    1.0,
                    2.0,
                    2.0,
                    3.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    2.0,
                    3.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    3.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        ).T
        assert (expansion == target).all()

    def test_im2row_with_stride(self):
        # input_shape = (1, 5, 5, 1)
        input_shape = (1, 1, 5, 5)
        img = np.array(
            [
                [1, 2, 3, 0, 1],
                [0, 1, 2, 3, 2],
                [3, 0, 1, 2, 3],
                [2, 3, 0, 1, 0],
                [1, 0, 3, 2, 1],
            ]
        )
        inputs = np.zeros((input_shape))
        # inputs[0, :, :, 0] = img
        inputs[0, 0, :, :] = img
        fh = fw = 3
        pad = 0
        stride = 2
        expansion = im2row(inputs=inputs, fh=fh, fw=fw, pad=pad, stride=stride)
        target = np.array(
            [
                [1, 2, 3, 0, 1, 2, 3, 0, 1],
                [3, 0, 1, 2, 3, 2, 1, 2, 3],
                [3, 0, 1, 2, 3, 0, 1, 0, 3],
                [1, 2, 3, 0, 1, 0, 3, 2, 1],
            ],
            dtype="float64",
        )
        assert (expansion == target).all()

    def test_im2row_loss_features(self):
        # 输入特征图形状为4x4
        # 过滤器形状为3x3
        # 填充为0， 滑动步长为2
        # 此时，过滤器只能在输入特征图上活动一次，输入特征图中局部感受野（即活动窗口）外的像素将会丢失
        fh = fw = 3
        pad = 0
        stride = 2
        expansion = im2row(self.inputs, fh, fw, pad, stride)
        # print('\n', 'expansion are:')
        # print(expansion)
        # print('')
        # print(expansion.shape)
        target = np.array([[1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0]])
        assert (expansion == target).all()

    def test_im2row_with_batch(self):
        # input_shape = (2, 2, 2, 3)
        input_shape = (2, 3, 2, 2)
        inputs = np.arange(1, 25).reshape(*input_shape)
        """
        第一批3个通道上的像素矩阵：
        [[1, 2],    [[5, 6],    [[9, 10],
         [3, 4]]     [7, 8]]     [11, 12]]
        第二批3个通道上的像素矩阵：
        [[13, 14],  [[17, 18],  [[21, 22],
         [15, 16]]   [19, 20]]   [23, 24]]
        ----------------------------------
        将统一通道上的局部感受野展开之后：
        第一批：            第二批：
        [[1, 5, 9],         [[13, 17, 21],
         [2, 6, 10],         [14, 18, 22],
         [3, 7, 11],         [15, 19, 23],
         [4, 8, 12]]         [16, 20, 24]]
        """
        print("\n", "inputs are:")
        print(inputs)
        print("")
        expansion = im2row(inputs, fh=1, fw=1, pad=0, stride=1)
        # print('\n')
        # print(expansion)
        # print('')
        target = np.array(
            [
                [1.0, 5.0, 9.0],  # 相同通道的像素都扩展到同一行
                [2.0, 6.0, 10.0],
                [3.0, 7.0, 11.0],
                [4.0, 8.0, 12.0],
                # ----------- 第一批
                [13.0, 17.0, 21.0],
                [14.0, 18.0, 22.0],
                [15.0, 19.0, 23.0],
                [16.0, 20.0, 24.0],
            ]
        )
        # ----------- 第二批
        assert (expansion == target).all()


class Testrow2im(TestCase):

    def test_row2im(self):
        expansion = np.array(
            [
                [1, 2, 3, 0, 1, 2, 3, 0, 1],
                [2, 3, 0, 1, 2, 3, 0, 1, 2],
                [0, 1, 2, 3, 0, 1, 2, 3, 0],
                [1, 2, 3, 0, 1, 2, 3, 0, 1],
            ]
        )
        inputs = row2im(expansion, (1, 1, 4, 4), 3, 3)
        img = inputs[0, 0, :, :]
        target = np.array([[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]])
        assert (img == target).all()


class TestConv2D(TestCase):

    def setUp(self):
        self.input_shape = (3, 4, 4)  # (10, 3, 4, 4)
        self.conv = Conv2D(
            filter_num=5,
            filter_size=(3, 3),
            activation="relu",
            initialization="he",
            input_shape=self.input_shape,
            pad=0,
            stride=1,
        )

    def test_init(self):
        assert self.conv.weights.shape == (5, 3, 3, 3)
        assert self.conv.biases.shape == (5,)
        # assert self.conv.output_shape == (10, 5, 2, 2)
        assert self.conv.output_shape == (5, 2, 2)

    def test__build(self):
        conv = Conv2D(5, (3, 3), "relu", "he", pad=0, stride=1)
        # input_shape = (10, 3, 4, 4)
        assert conv.built is False
        conv._build(self.input_shape)
        assert conv.built is True
        assert conv.oh == conv.ow == 2
        # assert conv.output_shape == (10, 5, 2, 2)
        assert conv.output_shape == (5, 2, 2)
        assert conv.weights.shape == (5, 3, 3, 3)
        assert conv.biases.shape == (5,)

    def test__build_raise_valueerror(self):
        conv = Conv2D(5, (3, 3), "relu", "he", pad=0, stride=1)
        input_shape = (4, 4)
        with pt.raises(ValueError):
            conv._build(input_shape)

    def test_build(self):
        conv = Conv2D(5, (3, 3), "relu", "he", pad=0, stride=1)
        # input_shape = (10, 3, 4, 4)
        assert conv.built is False
        conv.build(input_shape=self.input_shape)
        assert conv.built is True
        assert conv.oh == conv.ow == 2
        assert conv.output_shape == (5, 2, 2)
        assert conv.weights.shape == (5, 3, 3, 3)
        assert conv.biases.shape == (5,)

    def test_backward(self):
        inputs = np.random.randn(10, 3, 4, 4)
        outputs = self.conv.forward(inputs)
        ug = np.ones_like(outputs)
        grads = self.conv.backward(ug)
        ig, wg, bg = grads["i"], grads["w"], grads["b"]
        assert ig.shape == (10, 3, 4, 4)
        assert wg.shape == (5, 3, 3, 3)
        assert bg.shape == (5,)

    def test_backward_raise_backwardbeforeforward(self):
        inputs = np.random.randn(10, 3, 4, 4)
        # outputs = self.conv.forward(inputs)
        # ug = np.ones((100, 2, 2, 10))
        ug = np.ones((10, 5, 2, 2))
        with pt.raises(BackwardBeforeForward):
            self.conv.backward(ug)

    def test_forward(self):
        # filter_shape = (3, 3, 3, 3)  # (n, c, ih, iw)
        # input_shape = (3, 3, 5, 5)
        input_shape = (3, 5, 5)
        conv = Conv2D(
            filter_num=3,
            filter_size=(3, 3),
            activation="identity",
            initialization="he",
            pad=0,
            stride=1,
            input_shape=input_shape,
        )
        assert conv.biases.all() == 0.0

        inputs = np.random.randn(3, 3, 5, 5)
        outputs = conv.forward(inputs)
        assert outputs.shape == (3, 3, 3, 3)  # (n, fn, oh, ow)
        outputs = outputs.transpose(0, 2, 3, 1).reshape(3 * 3 * 3, 3)

        expansion = im2row(inputs, conv.fh, conv.fw, conv.pad, conv.stride)
        assert expansion.shape == (27, 27)  # 这里设置为可逆矩阵
        weight_reshaped = np.dot(np.linalg.inv(expansion), outputs)
        weights = weight_reshaped.T.reshape(3, 3, 3, 3)
        assert str(weights) == str(conv.weights)


class TestMaxPool2D(TestCase):
    def setUp(self):
        # self.input_shape = (1, 3, 4, 4)
        self.input_shape = (3, 4, 4)
        # self.output_shape = (1, 3, 2, 2)
        self.output_shape = (3, 2, 2)
        self.pool = MaxPool2D(
            pool_size=(2, 2), input_shape=self.input_shape, stride=None
        )

        self.inputs = np.array(
            [
                1,
                2,
                3,
                0,
                0,
                1,
                2,
                4,
                1,
                0,
                4,
                2,
                3,
                2,
                0,
                1,
                # ---------
                3,
                0,
                6,
                5,
                4,
                2,
                4,
                3,
                3,
                0,
                1,
                0,
                2,
                3,
                3,
                1,
                # ---------
                4,
                2,
                1,
                2,
                0,
                1,
                0,
                4,
                3,
                0,
                6,
                2,
                4,
                2,
                4,
                5,
            ],
            dtype="float64",
        )
        self.inputs = self.inputs.reshape((1, 3, 4, 4))
        # print('\n', 'inputs are:')
        # print(self.inputs)
        # print('')

    def test__build(self):
        pool = MaxPool2D((2, 2))
        assert pool.built is False
        input_shape = (3, 4, 4)
        pool._build(input_shape=input_shape)
        assert pool.built is True
        assert pool.oh == pool.ow == 2
        assert pool.output_shape == (3, 2, 2)

    def test__build_raise_valueerror(self):
        pool = MaxPool2D((2, 2))
        assert pool.built is False
        input_shape = (4, 4)
        with pt.raises(ValueError):
            pool.build(input_shape=input_shape)

    def test_build(self):
        pool = MaxPool2D((2, 2))
        assert pool.built is False
        input_shape = (3, 4, 4)
        pool.build(input_shape=input_shape)
        assert pool.built is True
        assert pool.oh == pool.ow == 2
        assert pool.output_shape == (3, 2, 2)

    def test_backward(self):
        ug = np.array([2, 4, 3, 4, 4, 6, 3, 3, 4, 4, 4, 6], dtype="float64")
        ug = ug.reshape(1, 3, 2, 2)
        self.pool.forward(self.inputs)
        grad = self.pool.backward(ug).get("i")
        # print('\n', 'grad are:')
        # print(grad)
        # print('')
        target = np.array(
            [
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                4,
                0,
                0,
                4,
                0,
                3,
                0,
                0,
                0,
                # ---------
                0,
                0,
                6,
                0,
                4,
                0,
                0,
                0,
                3,
                0,
                0,
                0,
                0,
                0,
                3,
                0,
                # ---------
                4,
                0,
                0,
                0,
                0,
                0,
                0,
                4,
                0,
                0,
                6,
                0,
                4,
                0,
                0,
                0,
            ],
            dtype="float64",
        ).reshape(1, 3, 4, 4)
        assert (grad == target).all()

    def test_forward(self):
        outputs = self.pool.forward(self.inputs)
        # print('\n', 'outputs are:')
        # print(outputs)
        # print('')
        target = np.array([2, 4, 3, 4, 4, 6, 3, 3, 4, 4, 4, 6], dtype="float64")
        assert (outputs.flatten() == target).all()


class TestFlatten(TestCase):
    """ fakeras.layer.Flatten的单元测试. """

    def setUp(self):
        # self.input_shape = (100, 1, 28, 28)
        self.input_shape = (1, 28, 28)
        self.flatten = Flatten(input_shape=self.input_shape)

    def test__build(self):
        flatten = Flatten()
        assert flatten.built is False
        flatten._build(input_shape=self.input_shape)
        assert flatten.built is True
        assert flatten.output_shape == (784,)

    def test_build(self):
        flatten = Flatten()
        assert flatten.built is False
        flatten.build(input_shape=self.input_shape)
        assert flatten.built is True
        assert flatten.output_shape == (784,)

    def test_backward(self):
        inputs = np.random.randn(1, 1, 28, 28)
        self.flatten.forward(inputs)
        ug = np.random.random(self.flatten.output_shape)
        grads = self.flatten.backward(ug)
        assert grads["i"].shape == inputs.shape

    def test_backward_raise_backwardbeforeforward(self):
        ug = np.random.random(self.flatten.output_shape)
        assert self.flatten.forward_done is False
        with pt.raises(BackwardBeforeForward):
            self.flatten.backward(ug)

    def test_forward(self):
        inputs = np.random.randn(100, *self.input_shape)
        outputs = self.flatten.forward(inputs)
        assert outputs.shape == (100, 784)


class TestLayerStack(TestCase):
    def setUp(self):
        pass
