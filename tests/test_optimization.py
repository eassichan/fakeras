# -*- coding: utf-8 -*-

import copy
import os
import sys
from unittest import TestCase

import numpy as np
import pytest as pt

from fakeras.layer import Dense
from fakeras.network import NeuralNetwork
from fakeras.optimization import GD, RMSProp

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))


@pt.fixture(scope="class")
def network(request):
    network = NeuralNetwork()
    network.add(Dense(10, "relu", "he", input_shape=(10,)))
    network.add(Dense(1, "identity", "he", input_shape=(10,)))
    network.loss_func_obj = "mse"
    network.metric_obj = "mae"
    x = np.random.randn(100, 10)
    y = network.predict(x)
    t = np.ones_like(y)
    network.loss(x, t)
    network.differentiate()
    request.cls.network = network


@pt.mark.usefixtures("network")
class TestGD(TestCase):
    """ fakeras.optimization.GD的单元测试. """

    def setUp(self):
        pass

    def test_update(self):
        net = self.network
        lr = 0.01

        old_w1 = copy.deepcopy(net[0].weights)
        old_b1 = copy.deepcopy(net[0].biases)
        old_w2 = copy.deepcopy(net[1].weights)
        old_b2 = copy.deepcopy(net[1].biases)

        gd = GD(net, lr)
        gd.update()

        w1 = net[0].weights
        w1 += gd.lr * net[0].grads["w"]
        assert str(w1) == str(old_w1)

        b1 = net[0].biases
        b1 += gd.lr * net[0].grads["b"]
        assert str(b1) == str(old_b1)

        w2 = net[1].weights
        assert w2.all()
        w2 += gd.lr * net[1].grads["w"]
        assert str(w2) == str(old_w2)

        b2 = net[1].biases
        b2 += gd.lr * net[1].grads["b"]
        assert str(b2) == str(old_b2)


@pt.mark.usefixtures("network")
class TestRMSProp(TestCase):
    """ fakeras.optimization.RMSProp的单元测试. """

    def setUp(self):
        pass

    def test_update(self):
        network = self.network
        beta = 0.9
        lr = 0.001

        old_w1 = copy.deepcopy(network[0].weights)
        old_b1 = copy.deepcopy(network[0].biases)
        old_w2 = copy.deepcopy(network[1].weights)
        old_b2 = copy.deepcopy(network[1].biases)

        rmsprop = RMSProp(network, beta, lr)
        rmsprop.update()

        w1 = network[0].weights
        w1 += (
            rmsprop.lr
            * network[0].grads["w"]
            / np.sqrt(rmsprop.accumulators[0]["w"] + rmsprop.epsilon)
        )
        assert str(w1) == str(old_w1)

        b1 = network[0].biases
        b1 += (
            rmsprop.lr
            * network[0].grads["b"]
            / np.sqrt(rmsprop.accumulators[0]["b"] + rmsprop.epsilon)
        )
        assert str(b1) == str(old_b1)

        w2 = network[1].weights
        assert w2.all()
        w2 += (
            rmsprop.lr
            * network[1].grads["w"]
            / np.sqrt(rmsprop.accumulators[1]["w"] + rmsprop.epsilon)
        )
        assert str(w2) == str(old_w2)

        b2 = network[1].biases
        b2 += (
            rmsprop.lr
            * network[1].grads["b"]
            / np.sqrt(rmsprop.accumulators[1]["b"] + rmsprop.epsilon)
        )
        assert str(b2) == str(old_b2)
