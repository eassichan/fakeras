# -*- coding: utf-8 -*-

import copy
import os
import sys
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pytest as pt

from fakeras.activation import ACTIVATIONS
from fakeras.exception import (
    BackwardBeforeForward,
    EmptyInputShape,
    LossFunctionNotFound,
    MetricNotFound,
    OptimizerNotFound,
)
from fakeras.layer import Dense
from fakeras.loss import LOSSES
from fakeras.loss import CategoricalCrossEntropy as CCE
from fakeras.metric import METRICS, CategoricalAccuracy
from fakeras.network import NeuralNetwork
from fakeras.optimization import GD, OPTIMIZERS

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


class TestNeuralNetwork(TestCase):
    """ fakeras.network.NeuralNetwork的单元测试. """

    def setUp(self):
        self.empty_network = NeuralNetwork()

        self.softmax_network = NeuralNetwork()
        self.input_shape = (100,)
        self.softmax_network.add(
            Dense(
                100,
                activation="softmax",
                initialization="he",
                input_shape=self.input_shape,
            )
        )
        self.softmax_network.build()
        self.softmax_network.compile(optimizer="gd", loss_func="cce", metric="acc")

        self.identity_network = NeuralNetwork()
        self.identity_network.add(
            Dense(
                100,
                activation="identity",
                initialization="he",
                input_shape=self.input_shape,
            )
        )
        self.identity_network.build()
        self.identity_network.compile(optimizer="gd", loss_func="mse", metric="mae")

        self.deep_network = NeuralNetwork()
        self.deep_network.add(
            Dense(10, activation="relu", initialization="he", input_shape=(10,))
        )
        self.deep_network.add(Dense(9, activation="relu", initialization="he"))
        self.deep_network.add(Dense(8, activation="relu", initialization="he"))
        self.deep_network.add(Dense(7, activation="relu", initialization="he"))
        self.deep_network.add(Dense(6, activation="relu", initialization="he"))

        self.deep_network.add(Dense(5, activation="relu", initialization="he"))
        self.deep_network.add(Dense(4, activation="relu", initialization="he"))
        self.deep_network.add(Dense(3, activation="relu", initialization="he"))
        self.deep_network.add(Dense(2, activation="relu", initialization="he"))
        self.deep_network.add(
            Dense(
                1,
                activation="identity",
                initialization="he",
                input_shape=self.input_shape,
            )
        )
        self.deep_network.build()
        self.deep_network.compile(optimizer="gd", loss_func="mse", metric="mae")

    def test__getitem__(self):
        self.empty_network.add(
            Dense(10, activation="relu", initialization="he", input_shape=(100,))
        )
        assert isinstance(self.empty_network[0], Dense)

    def test__getitem__raise_indexerror(self):
        self.empty_network.add(
            Dense(10, activation="relu", initialization="he", input_shape=(100,))
        )
        assert len(self.empty_network) == 1
        with pt.raises(IndexError) as e:
            layer = self.empty_network[1]
        assert e.value.args[0] == "Layer's index out of range."

    def test__iter__(self):
        count = 0
        for layer in self.deep_network:
            count += 1
            assert isinstance(layer, Dense)
        assert count == 10

    def test__len__(self):
        assert len(self.deep_network) == 10

    def test__reversed__(self):
        counter = 1
        for layer in reversed(self.deep_network):
            assert isinstance(layer, Dense)
            assert layer.neuron_num == counter
            counter += 1

    def test_loss_func_obj(self):
        self.empty_network.loss_func_obj = "cce"
        assert isinstance(self.empty_network.loss_func_obj, CCE)

        self.empty_network.loss_func_obj = CCE()
        assert isinstance(self.empty_network.loss_func_obj, CCE)

    def test_loss_func_obj_raise_lossfunctionnotfound(self):
        func_name = "loss"
        assert func_name not in LOSSES
        with pt.raises(LossFunctionNotFound):
            self.empty_network.loss_func_obj = func_name

    def test_loss_func_obj_raise_typeerror(self):
        func_name = 10
        with pt.raises(TypeError):
            self.empty_network.loss_func_obj = func_name

    def test_metric_obj(self):
        self.empty_network.metric_obj = "acc"
        assert isinstance(self.empty_network.metric_obj, CategoricalAccuracy)

        self.empty_network.metric_obj = CategoricalAccuracy()
        assert isinstance(self.empty_network.metric_obj, CategoricalAccuracy)

    def test_metric_obj_raise_metricnotfound(self):
        metric_name = "mse"
        assert metric_name not in METRICS
        with pt.raises(MetricNotFound):
            self.empty_network.metric_obj = metric_name

    def test_metric_obj_raise_typeerror(self):
        metric_name = 3.1415
        with pt.raises(TypeError):
            self.empty_network.metric_obj = metric_name

    def test_optimizer_obj(self):
        self.empty_network.optimizer_obj = "gd"
        assert isinstance(self.empty_network.optimizer_obj, GD)

        self.empty_network.optimizer_obj = GD(network=self.empty_network, lr=0.001)
        assert isinstance(self.empty_network.optimizer_obj, GD)

    def test_optimizer_obj_raise_optimizernotfound(self):
        optimizer_name = "convex_optimization"
        assert optimizer_name not in OPTIMIZERS
        with pt.raises(OptimizerNotFound):
            self.empty_network.optimizer_obj = optimizer_name

    def test_optimizer_obj_raise_typeerror(self):
        optimizer_name = 1e-10
        with pt.raises(TypeError):
            self.empty_network.optimizer_obj = optimizer_name

    def test__batching_use_every_sample(self):
        sample_count = 60000
        batch_size = 100
        use_every_sample = True
        idxes = [
            i
            for i in self.empty_network._batching(
                sample_count, batch_size, use_every_sample
            )
        ]
        idx_set = set(idxes)
        assert len(idxes) == 600
        assert len(idx_set) == len(idxes)

    def test__batching_dont_use_every_sample(self):
        sample_count = 100
        batch_size = 10
        use_every_sample = False
        idxes = [
            i
            for i in self.empty_network._batching(
                sample_num=sample_count,
                batch_size=batch_size,
                use_every_sample=use_every_sample,
            )
        ]
        idx_set = set(idxes)
        assert len(idx_set) != len(idxes)

    def test__collect_historical_data(self):
        histroies = []
        epoch = 6
        result_train = dict(loss=0.02134, metric=0.87)
        result_val = dict(loss=0.01234, metric=0.67)
        metric = "acc"
        time_used = 12.25698
        history = self.empty_network._collect_historical_data(
            histroies, epoch, result_train, result_val, metric, time_used
        )
        assert [
            k in history
            for k in {"epoch", "loss", "acc", "loss_val", "acc_val", "time_used"}
        ]

        result_val = None
        history = self.empty_network._collect_historical_data(
            histroies, epoch, result_train, result_val, metric, time_used
        )
        assert [k in history for k in {"epoch", "loss", "acc", "time_used"}]

    def test_add(self):
        self.empty_network.add(
            Dense(10, activation="relu", initialization="he", input_shape=(100,))
        )
        assert len(self.empty_network) == 1
        assert isinstance(self.empty_network.layers[0], Dense)

    def test_build(self):
        self.empty_network.add(
            Dense(64, activation="relu", initialization="he", input_shape=(784,))
        )
        self.empty_network.add(Dense(10, activation="relu", initialization="he"))
        self.empty_network.build()
        assert len(self.empty_network) == 2
        assert self.empty_network[0].weights.shape == (784, 64)
        assert self.empty_network[1].weights.shape == (64, 10)

    def test_build_raise_emptyinputshape(self):
        self.empty_network.add(Dense(10, activation="relu", initialization="he"))
        with pt.raises(EmptyInputShape):
            self.empty_network.build()

    def test_compile(self):
        optimizer = "gd"
        loss_func = "cce"
        metric = "acc"
        self.empty_network.compile(optimizer, loss_func, metric)
        assert isinstance(self.empty_network.optimizer_obj, GD)
        assert isinstance(self.empty_network.loss_func_obj, CCE)
        assert isinstance(self.empty_network.metric_obj, CategoricalAccuracy)

    def test_differentiate(self):
        x = np.random.random((100, 100))
        t = np.identity(100)
        y = self.identity_network.predict(x)
        self.identity_network.loss(x, t)
        self.identity_network.differentiate()

        assert len(self.identity_network) == 1
        layer = self.identity_network[-1]  # 网络只有一层
        grads = layer.grads
        ig, wg, bg = grads["i"], grads["w"], grads["b"]
        ug = self.identity_network.loss_func_obj.backward()
        assert str(ug - layer.biases) == str(np.dot(np.linalg.inv(x.T), wg))
        assert str(ug - layer.biases) == str(np.dot(ig, np.linalg.inv(layer.weights.T)))

    def test_dfiferentiate_raise_backwardbeforeforeward(self):
        self.empty_network.loss_func_obj = "cce"
        with pt.raises(BackwardBeforeForward):
            self.empty_network.differentiate()

    def test_evaluate(self):
        inputs = np.random.random((100, 100))
        targets = np.identity(100)  # 单位矩阵
        result = self.softmax_network.evaluate(inputs, targets)

        assert self.softmax_network.weight_decay_lambda == 0.0
        outputs = self.softmax_network.predict(inputs)
        assert result["loss"] == self.softmax_network.loss_func_obj(outputs, targets)
        assert result["metric"] == self.softmax_network.metric_obj(outputs, targets)

    def test_fit(self):
        inputs = np.random.random((100, 100))
        targets = np.identity(100)
        loss_func = "cce"
        metric = "acc"
        optimizer = "gd"
        batch_size = 20
        epochs = 10
        lr = 0.01
        use_every_sample = False
        histories = self.softmax_network.fit(
            inputs=inputs,
            targets=targets,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            use_every_sample=use_every_sample,
            verbose=False,
        )
        assert len(histories) == 10

    def test_loss(self):
        inputs = np.random.random((100, 100))
        targets = np.identity(self.input_shape[0])
        e = self.softmax_network.loss(inputs, targets)
        outputs = self.softmax_network.predict(inputs)
        assert e == self.softmax_network.loss_func_obj(outputs, targets)

        self.softmax_network.weight_decay_lambda = 0.001
        e = self.softmax_network.loss(inputs, targets)
        assert len(self.softmax_network) == 1
        weights_decay = self.softmax_network.weight_decay_lambda * np.sum(
            self.softmax_network[-1].weights ** 2
        )
        weights_decay /= self.input_shape[0]
        assert e - weights_decay == self.softmax_network.loss_func_obj(outputs, targets)

    def test_predict(self):
        rng = np.random.RandomState(seed=42)
        inputs = rng.randn(*self.input_shape)
        outputs = self.softmax_network.predict(inputs)
        assert len(self.softmax_network) == 1

        last_layer = self.softmax_network[-1]
        w = last_layer.weights
        b = last_layer.biases
        z = np.dot(inputs, w) + b
        y = last_layer.activator(z)
        assert (outputs == y).all()


class TestNetworkForRegression(TestCase):
    """ Functional test for regression by using neural network. """

    def setUp(self):
        pass

    def test_network_for_regression(self):
        # 先基于y=2x+3生成一些用于回归的数据：
        x = np.linspace(-1, 1, 300)
        y = 2 * x + 3 + np.random.randn(*x.shape) * 0.5

        # 矩阵化原始数据：
        x_vector = x.copy()
        x = x.reshape(len(x), -1)
        t = y.reshape(len(y), -1)
        # print('\n', x.shape)
        # print(t.shape)

        network = NeuralNetwork()
        network.add(
            Dense(1, activation="identity", initialization="he", input_shape=(1,))
        )
        # network.build()
        network.compile(optimizer="rmsprop", loss_func="mse", metric="mae")
        network.fit(
            inputs=x,
            targets=t,
            batch_size=100,
            epochs=200,
            lr=0.01,
            use_every_sample=True,
            verbose=False,
        )
        weight = network[-1].weights[-1][-1]
        bias = network[-1].biases[-1][-1]
        assert np.abs(2 - weight) < 0.2
        assert np.abs(3 - bias) < 0.2
        # print('\n')
        # print(weight, bias)

        # 绘制模型曲线：
        print("\n", weight, bias)
        x_min, x_max = x_vector.min(), x_vector.max()
        y_hat = np.dot(x, weight) + bias
        y_min, y_max = y_hat.min(), y_hat.max()
        plt.plot([x_max, x_min], [y_max, y_min], color="r")
        plt.scatter(x, y)
        file_name = os.path.join(
            os.path.dirname(__file__), "reports", "test_network_for_regression.png"
        )
        plt.savefig(file_name)


class TestEvaluate(TestCase):
    """ fakeras.network.NeuralNetwork.evaluate方法的详细测试. """

    def setUp(self):
        pass

    @pt.mark.usefixtures("bin_classifier")
    def test_evaluate_with_binaryaccuracy(self):
        x = np.random.randn(100, 100)
        t = np.ones((100, 1))
        t[t % 3 == 0] = 0
        loss_1, metric_1 = self.model.evaluate(x, t, batch_size=len(x))
        loss_2, metric_2 = self.model.evaluate(x, t, batch_size=64)
        assert (loss_1, metric_1) == (loss_2, metric_2)

    @pt.mark.usefixtures("multi_classifier")
    def test_evaluate_with_categoricalaccuracy(self):
        x = np.random.randn(100, 100)
        t = np.identity(100)
        loss_1, metric_1 = self.model.evaluate(x, t, batch_size=len(x))
        loss_2, metric_2 = self.model.evaluate(x, t, batch_size=32)
        assert (loss_1, metric_1) == (loss_2, metric_2)

    @pt.mark.usefixtures("linear_regressor")
    def test_evaluate_with_meanabsoluteerror(self):
        x = np.linspace(-1, 1, 300)
        t = 2 * x + 3 + np.random.randn(*x.shape) * 0.5
        x.resize(len(x), 1)
        t.resize(len(t), 1)
        # print('\n', x.shape)
        # print('\n', y.shape)
        loss_1, metric_1 = self.model.evaluate(x, t, batch_size=len(x))
        loss_2, metric_2 = self.model.evaluate(x, t, batch_size=32)
        assert (loss_1, metric_1) == (loss_2, metric_2)


@pt.fixture()
def bin_classifier(request):
    model = NeuralNetwork()
    model.add(Dense(16, "relu", "he", input_shape=(100,)))
    model.add(Dense(1, "sigmoid", "he"))
    model.build()
    model.compile(optimizer="gd", loss_func="bce", metric="bin_acc")
    request.cls.model = model


@pt.fixture()
def multi_classifier(request):
    model = NeuralNetwork()
    model.add(Dense(100, "relu", "he", input_shape=(100,)))
    model.add(Dense(100, "softmax", "he"))
    model.build()
    model.compile("gd", "cce", "acc")
    request.cls.model = model


@pt.fixture()
def linear_regressor(request):
    model = NeuralNetwork()
    model.add(Dense(1, "identity", "he", input_shape=(1,)))
    model.build()
    model.compile("gd", "mse", "mae")
    request.cls.model = model
