# -*- coding: utf-8 -*-

import os
import sys
from unittest import TestCase

import numpy as np
import pytest as pt

from fakeras.activation import ReLU
from fakeras.initialization import he_normalize, xavier_uniformize

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))


def tanh(x):
    val1 = np.exp(x) - np.exp(-x)
    val2 = np.exp(x) + np.exp(-x)
    y = val1 / val2
    return y


def relu(x):
    if isinstance(x, int) or isinstance(x, float):
        if x >= 0:
            y = x
        else:
            y = 0
    if isinstance(x, np.ndarray):
        y = x.copy()
        y[x < 0] = 0
    else:
        raise TypeError
    return y


class LearningInitialization(TestCase):
    """ 学习权重初始化策略.

    参考资料：https://cloud.tencent.com/developer/article/1522070

    1. 为什么需要初始化权重；（防止梯度爆炸和梯度消失，从而保证能够正确学习，
       即避免学习提前终止或者加快学习的收敛速度）
    2. 梯度爆炸；
    3. 梯度消失；
    4. 在没有激活函数的情况下，避免梯度爆炸和梯度消失；
    5. 在激活函数为tanh或softsign的情况下，使用xavier策略避免梯度爆炸和梯度消失；
    6. 在激活函数为relu的情况下，使用he kaiming策略避免梯度爆炸和梯度消失.
    """

    def test_gradient_explosion(self):
        """ 梯度爆炸. """
        x = np.random.randn(512, 512)
        i = 0
        for i in range(10000):
            w = np.random.randn(512, 512)
            x = np.dot(w, x)
            if np.isnan(x.mean()) and np.isnan(x.std()):
                break
        assert 220 <= i <= 230  # 大概在第225轮之间发生梯度爆炸

    def test_gradient_vanishing(self):
        """ 梯度消失. """
        x = np.random.randn(512, 512)
        i = 0
        for i in range(1000):
            w = np.random.randn(512, 512) * 0.01
            x = np.dot(w, x)
            if x.mean() == 0.0 and x.std() == 0.0:
                break
        assert 490 <= i <= 500  # 大约在第496轮发生梯度消失

    def test_output_stddev_with_nomal_dist(self):
        """ 在使用标准正态分布初始化权重参数和没有激活函数的情况下，
            模拟神经网络每一层输出的均值、方差以及标准差.
        """
        # 1). 当输入只有一个元素的情况下，每一层输出的均值、方差和标准差分别为0, 1, 1：
        n = 1
        mean, var = 0.0, 0.0
        epoch = 10000
        for i in range(epoch):
            w = np.random.randn(n)
            x = np.random.randn(n)
            y = w * x
            mean += y.mean()
            var += np.mean(np.power(y, 2))
        mean /= epoch
        std = np.sqrt(var / epoch)
        assert np.round(mean) == 0.0 and np.round(std) == 1.0

        # 2). 当输入向量有n个元素时，每一层输出的均值、方差和标准差分别为0, n, sqrt(n)：
        n = 512
        x = np.random.randn(n)
        mean, var = 0.0, 0.0
        epoch = 100
        for i in range(epoch):
            w = np.random.randn(n, n)
            y = np.dot(w, x)
            mean += y.mean()
            var += np.mean(np.power(y, 2))
        mean /= epoch
        var /= epoch
        std = np.sqrt(var)
        assert np.round(mean) == 0.0
        assert -100 <= var - n <= 100  # 理论上方应该差约等于输出向量中的元素个数n
        assert -2 <= np.sqrt(n) - std <= 2

    def test_avoid_explosion_and_vanishing(self):
        """ 在没有激活函数以及使用标准正态分布的情况下，
            避免梯度爆炸和梯度消失.

            由上一个测试知道：
            1）当输出只有一个元素的时候，输出的均值、方差和标准差分别为0, 1, 1；
            2）当输出有n个元素的时候，输出的均值、方差和标准差就变为0，n，sqrt(n)；
            3）可以看到当输出的标准差为1的时候，即时训练周期epoch比较大（epoch=1000），
               训练过程中也没有发生梯度爆炸或者梯度消失，因此，在输出有n个元素的情况下，
               可以尝试限制各层的输出标准差为1，从而避免梯度爆炸和梯度消失，具体的做法
               是：对权重张量缩小sqrt(n)倍.
        """
        mean, var = 0.0, 0.0
        epoch = 1000
        n = 512
        x = np.random.randn(n)
        for i in range(epoch):
            w = np.random.randn(n, n) * np.sqrt(1 / 512)  # 缩放权重
            y = np.dot(w, x)
            mean += y.mean()
            var += np.mean(np.power(y, 2))
        mean /= epoch
        var /= epoch
        std = np.sqrt(var)
        assert np.round(mean) == 0.0
        assert np.round(var) == 1.0
        assert np.round(std) == 1.0
        
    def test_xavier_with_tanh(self):
        """ TODO: 这个实验还是不太明白. """
        mean, var = 0.0, 0.0
        m = n = 512
        x = np.random.randn(n)
        epoch = 100
        for i in range(epoch):
            w = np.random.uniform(-1, 1) * np.sqrt(6 / (m + n))
            y = tanh(x)
            mean += y.mean()
            var += np.mean(np.power(y, 2))
        mean /= epoch
        var /= epoch
        std = np.sqrt(var)
        print('')
        print(mean)
        print(var)
        print(std)
    
    def test_kaming_with_relu(self):
        mean, var = 0.0, 0.0
        m = n = 512
        x = np.random.randn(n)
        epoch = 100
        for i in range(epoch):
            w = np.random.randn(m, n) * np.sqrt(2 / 512)
            y = relu(np.dot(w, x))
            mean += y.mean()
            var += np.mean(np.power(y, 2))
        mean /= epoch
        var /= epoch
        std = np.sqrt(var)
        assert not np.round(mean) == 0.0  # 这里输出的均值并不为0，可能跟relu抹去了x<0那部分的值有关 TODO: 不太明白
        assert np.round(var) == 1.0
        assert np.round(std) == 1.0


class TestHeNormalize(TestCase):
    """ fakeras.initialization.he_normalize函数的单元测试. """

    def test_he_normalize_with_relu(self):
        """ 结合ReLU激活函数来测试何凯明初始化策略. """
        mean = 0.0
        var = 0.0
        epochs = 100
        for i in range(epochs):
            relu = ReLU()

            x = np.random.randn(512, 10)
            # w = np.random.randn(512, 512)
            w = he_normalize(weight_shape=(512, 512))
            b = np.zeros_like(x)
            y = relu(np.dot(w, x) + b)
            mean += np.mean(y)
            var += np.mean(np.square(y))
        mean /= epochs
        std = np.sqrt(var / epochs)
        assert np.abs(mean / 256) <= 1e-2  # y的均值应约等于0
        assert np.abs(std - 1) <= 1e-2  # y的标准差应约等于1


class TestXavierUniformize(TestCase):
    """ fakeras.initialization.xavier_uniformize函数的单元测试. """

    def test_uniformize(self):
        epochs = 100

        # 使用均匀分布(-sqrt(3), sqrt(3))分别初始化权重w和输入x，那么w和x各自的均值和方差都为0和1，
        # 所以w与x的乘积的均值和方差仍分别为0和1.
        mean = 0.0
        var = 0.0
        for i in range(epochs):
            w = np.random.uniform(-np.sqrt(3), np.sqrt(3), (512, 512))
            x = np.random.uniform(-np.sqrt(3), np.sqrt(3), (512, 10))
            b = np.zeros_like(x)
            y = np.dot(w, x) + b
            mean += np.mean(y)
            var += np.mean(np.square(y))
        mean = mean / epochs
        var = var / epochs
        # print('')
        # print(mean, var)
        assert np.abs(mean) <= 0.2  # 理论上均值应该接近0
        assert 510 <= np.round(var) < 515  # 512个w和x的元素乘积相加，其方差应该接近512

        # 使用均匀分布(-1, 1)分别初始化权重w和输入x，那么w和x各自的均值和方差都为0和1/3，
        # 所以w与x的乘积的方差为(1/3)*(1/3)=1/9.
        mean = 0.0
        var = 0.0
        for i in range(epochs):
            w = np.random.uniform(-1, 1, (512, 512))
            x = np.random.uniform(-1, 1, (512, 10))
            b = np.zeros_like(x)
            y = np.dot(w, x) + b
            mean += np.mean(y)
            var += np.mean(np.square(y))
        mean = mean / epochs
        var = var / epochs
        assert np.abs(mean) <= 0.1
        assert 0 < np.abs(var - 512 / 9) <= 0.5

    def test_xavier_uniformize(self):
        """ 测试Xavier初始化策略. """
        epochs = 1000
        mean = 0.0
        var = 0.0
        for _ in range(epochs):
            w = xavier_uniformize(weight_shape=(512, 512))
            x = np.random.uniform(-np.sqrt(3), np.sqrt(3), (512, 10))  # var=1
            # x = np.random.uniform(-np.sqrt(1), np.sqrt(1), (512, 10))  # var=1/3
            b = np.zeros_like(x)
            z = np.dot(w, x) + b
            mean += z.mean()
            var += np.square(z).mean()
        mean /= epochs
        var /= epochs
        assert np.abs(mean) <= 1e-2  # mean应该约等于0
        assert 0.9 < var < 1.1  # var应该约等于1