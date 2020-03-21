# -*- coding: utf-8 -*-

"""
fakeras.network
~~~~~~~~~~~~~~~

本模块实现了神经网络的完整实现，包括网络的构建，网络的训练，网络的评估，以及使用网络进行
预测等核心功能.
"""

import math
import time
import warnings

import numpy as np

from .exception import (
    FitBeforeCompile,
    LossFunctionNotFound,
    MetricNotFound,
    OptimizerNotFound,
    validate_data_type,
)
from .layer import Layer
from .loss import LOSSES as losses
from .loss import Loss
from .metric import METRICS as metrics
from .metric import Metric
from .optimization import OPTIMIZERS as optimizers
from .optimization import Optimizer

warnings.filterwarnings("error")


class NeuralNetwork:
    """ 神经网络.

    Attributes
    ----------
    _loss_func_obj: a subclass instance of fakeras.loss.Loss
        网络的损失函数，在训练网络的时候才会被赋值.
    _metric_obj: a subclass instance of fakeras.metric.Metric
        需要在网络训练时监控的指标，例如分类任务的分类精度，回归任务的平均绝对误差等等，
        同样在训练网络的时候才会被setter赋值.
    _optimizer_obj: a subclass instance of fakeras.optimization.optimizer_obj
        网络选用的优化器，用于最小化损失函数，也是在训练网络的时候才会被setter赋值.
    layers: list of subclass instance of fakeras.layer.Layer
        神经网络中所有的层.
    metric_name: str
        监控指标名.
    weight_decay_lambda: float
        权重衰减指数，作为控制L2正则化的超参数.

    Methods
    -------
    __getitem__:
        将神经网络看作是层的容器，提供按索引号访问层的功能.
    __iter__:
        基于神经网络转化为层的迭代器.
    __len__:
        将神经网络看作是层的容器之后，容器的长度就是神经网络的深度.
    loss_func_obj:
        用于训练网络时便捷地绑定、读取所选用的损失函数.
    metric_obj:
        用于训练网络时便捷地绑定、读取所选用的监控指标.
    optimizer_obj:
        用于训练网络时便捷地绑定、读取所选用的优化器.
    _batching:
        对训练集进行随机打乱、批次划分.
    _collect_historical_data:
        收集训练网络时每一轮epoch的损失误差以及监控指标值，这些数据被当做历史数据.
    _print_historical_data:
        将收集到的历史数据打印到终端上.
    add:
        将层或者计算节点添加到网络.
    build:
        将神经网络的各层连接起来.
    compile:
        设置优化器，损失函数和监控指标，与Keras一致.
    differentiate:
        从输出层开始，到第一个隐藏层为止，反向地求解损失函数对网络各层的梯度.
    evaluate:
        评估网络，评估指标包括损失误差和监控指标值.
    fit:
        训练网络，绑定损失函数，监控指标以及优化器，进而拟合训练集获取最优参数，并根据需要
        在验证集上验证模型性能.
    loss:
        求解总的损失误差，这里的损失误差是在损失函数的结果上加上正则化惩罚项之后的结果.
    predict:
        使用网络进行预测，也可以理解成对网络进行正向传播.
    train:
        训练神经网络，即执行网络的正向传播、反向传播以及使用优化器来更新每一层的参数.
    """

    def __init__(self):
        self._loss_func_obj = None
        self._metric_obj = None
        self._metric_name = None
        self._optimizer_obj = None

        self.layers = []
        self.weight_decay_lambda = 0.0

    def __getitem__(self, index):
        if index not in range(-len(self), len(self)):
            raise IndexError("Layer's index out of range.")
        return self.layers[index]

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    @property
    def loss_func_obj(self):
        return self._loss_func_obj

    @loss_func_obj.setter
    def loss_func_obj(self, loss_func):
        if isinstance(loss_func, str):
            loss_cls = losses.get(loss_func, None)
            if loss_cls is None:
                err_msg = "Loss function '%s' not found." % loss_func
                raise LossFunctionNotFound(err_msg)
            self._loss_func_obj = loss_cls()
        elif isinstance(loss_func, Loss):
            self._loss_func_obj = loss_func
        else:
            err_msg = (
                "Expect to receive loss-function names or loss-function objects, "
                "but %s is found." % (type(loss_func))
            )
            raise TypeError(err_msg)

    @property
    def metric_obj(self):
        return self._metric_obj

    @metric_obj.setter
    def metric_obj(self, metric):
        if isinstance(metric, str):
            metric_cls = metrics.get(metric, None)
            if metric_cls is None:
                err_msg = "Metric '%s' not found." % metric
                raise MetricNotFound(err_msg)
            self._metric_obj = metric_cls()
            self._metric_name = metric
        elif isinstance(metric, Metric):
            self._metric_obj = metric
            self._metric_name = type(metric).__name__
        else:
            err_msg = (
                "Expect to receive metric names or metric objects, "
                "but %s is found." % (type(metric))
            )
            raise TypeError(err_msg)

    @property
    def optimizer_obj(self):
        return self._optimizer_obj

    @optimizer_obj.setter
    def optimizer_obj(self, optimizer):
        if isinstance(optimizer, str):
            optimizer_cls = optimizers.get(optimizer, None)
            if optimizer_cls is None:
                err_msg = "optimizer '%s' not found." % optimizer
                raise OptimizerNotFound(err_msg)
            self._optimizer_obj = optimizer_cls(network=self)
        elif isinstance(optimizer, Optimizer):
            self._optimizer_obj = optimizer
        else:
            err_msg = (
                "Expect to receive optimizer names or optimizer objects, "
                "but %s is found." % (type(optimizer))
            )
            raise TypeError(err_msg)

    def _batching(self, sample_num, batch_size, shuffle=True, use_every_sample=True):
        """ 对训练集划分批次，然后按需打乱批次顺序.

        Parameters
        ----------
        sample_num: int
            样本总数.
        batch_size: int
            对训练集进行分批时，每一个批次所包含的样本数.
        use_every_sample: boolean
            在随机梯度下降算法中，因为用来训练的样本是随机挑选的，所以在同一轮训练里某些样本
            可能会被训练多次，而有些样本则没有得到训练. 而为了使优化算法在每一轮都可以遍历所
            有的样本，这里提供了一种解决办法：在每一轮训练当中，先将训练集洗牌打乱，然后一个
            批次接一个批次地训练.
            因此，当use_every_sample设置为False时，可理解为使用与随机梯度下降相同的训练策略；
            当use_every_sample设置为True时，训练时将会遍历所有的样本.

        Yields
        ------
        i: int
            打乱批次顺序后的批次序号.
        """
        batches = math.ceil(sample_num / batch_size)  # 总批数
        if not use_every_sample:
            batch_idxs = np.random.choice(batches, batches)
        else:
            batch_idxs = np.arange(batches)
            if shuffle:
                np.random.shuffle(batch_idxs)
        for i in batch_idxs:
            yield i

    def _collect_historical_data(self, histories, epoch, result_train,
                                 result_val, metric, time_used):
        """ 收集每一轮训练的历史数据.

        Parameters
        ----------
        histories: list of dict
            保存训练集经过每一轮训练后的损失误差和监控指标值，如果训练时传入了验证集
            （即validation_data），则会同时保存有关验证集的损失误差和监控指标值.
        epoch: int
            训练轮数，训练集中所有的样本都已经完成训练所经过的时间周期称为一个epoch.
        result_train: dict
            训练集经过一轮训练后的评估结果，即训练集的损失误差和监控指标值.
        result_val: dict
            如果在训练网络时（即调用fit方法时）传入了验证集（validation_data），那么
            result_val代表验证集经过一轮训练后的评估结果.
        metric: str
            监控指标名.
        time_used: float
            训练所用时间.

        Returns
        -------
        history: dict
            本轮训练的历史数据.
        """
        history = {
            "epoch": epoch,
            "loss": result_train["loss"],
            metric: result_train["metric"],
        }

        if result_val is not None:
            history["loss_val"] = result_val["loss"]
            history[metric + "_val"] = result_val["metric"]

        history["time_used"] = time_used
        histories.append(history)
        return history

    def _print_historical_data(self, epochs, verbose, history, metric):
        """ 打印历史数据到终端控制台.

        Parameters
        ----------
        epochs: int
            训练轮数，训练集中所有的样本都已经完成训练所经过的时间周期称为一个epoch.
        verbose: boolean
            是否将训练的历史数据打印到终端控制台.
        history: dict
            本轮训练的历史数据.
        metric: str
            监控指标名.
        """
        metric_name = metric
        if verbose:
            epoch = history["epoch"] + 1
            loss = history["loss"]
            metric = history[metric_name]
            loss_val = history.get("loss_val", None)
            metric_val = history.get(metric_name + "_val", None)
            time_used = history["time_used"]

            print("Epoch %s/%s - " % (epoch, epochs), end="")
            print("loss: %f - %s: %f - " % (loss, metric_name, metric), end="")
            if (loss_val, metric_val) != (None, None):
                print(
                    "loss_val: %f - %s: %f - "
                    % (loss_val, metric_name + "_val", metric_val),
                    end="",
                )
            print("time used: %ss." % time_used)

    def add(self, layer):
        validate_data_type(Layer, "layer", layer)
        self.layers.append(layer)

    def build(self):
        last_output_shape = self[0].output_shape
        for layer in self:
            layer.build(input_shape=last_output_shape)
            last_output_shape = layer.output_shape

    def compile(self, optimizer, loss_func, metric):
        self.optimizer_obj = optimizer
        self.loss_func_obj = loss_func
        self.metric_obj = metric

    def differentiate(self):
        ug = self.loss_func_obj.backward()  # upstream gradient, 上游梯度
        for layer in reversed(self.layers):
            grads = layer.backward(
                upstream_gradient=ug,
                weight_decay_lambda=self.weight_decay_lambda
            )
            ug = grads["i"]

    def evaluate(self, inputs, targets, batch_size=100):
        """ 评估网络的性能.

        Parameters
        ----------
        inputs: numpy.ndarray
            特征张量.
        targets: numpy.ndarray
            目标张量.

        Returns
        -------
        result: dict
            评估结果，包括损失误差以及训练监控指标值.
        """
        losses = 0
        metrics = 0
        for i in self._batching(inputs.shape[0], batch_size, shuffle=False):
            start = i * batch_size
            stop = (i + 1) * batch_size
            x = inputs[start:stop]
            t = targets[start:stop]
            scale = len(x) / inputs.shape[0]  # 对损失误差和监控指标进行加权平均的权重（注意分批大小可能并不是完全相同）

            y = self.predict(x)
            loss = self.loss(x, t)
            metric = self.metric_obj(y, t)
            losses += loss * scale
            metrics += metric * scale
        result = dict(loss=losses, metric=metrics)
        return result

    def fit(
        self,
        inputs,
        targets,
        batch_size=100,
        epochs=10,
        lr=0.01,
        validation_data=None,
        verbose=True,
        use_every_sample=True,
        weight_decay_lambda=0.0,
    ):
        """ 训练神经网络.

        Parameters
        ----------
        inputs: numpy.ndarray
            训练集的特征张量.
        targets: numpy.ndarray
            训练集的目标张量，即分类任务中的标签，回归任务中的目标值.
        batch_size: int
            对训练集进行分批时，每一个批次所包含的样本数.
            另外，每一个批次的所有样本都得到训练所经过的时间周期称为一个iteration.
        epochs: int
            训练轮数，训练集中所有的样本都已经完成训练所经过的时间周期称为一个epoch.
        lr: float
            learning rate，传入优化器的学习率，又称为更新步长.
        validation_data: tuple
            验证数据，其结构是一对特征和目标，即（特征张量，目标张量）.
        verbose: boolean
            是否将训练的历史数据打印到终端控制台.
        use_every_sample: boolean
            在随机梯度下降算法中，因为用来训练的样本是随机挑选的，所以在同一轮训练里某些样本
            可能会被训练多次，而有些样本则没有得到训练. 而为了优化算法每一轮都可以遍历所有的
            样本，这里提供了一种解决办法：在每一轮训练当中，先将训练集随机打乱，然后一个批次
            接一个批次地训练.
            因此，当use_every_sample设置为False时，可理解为使用与随机梯度下降相同的训练策略；
            当use_every_sample设置为True时，训练时将会遍历所有的样本.
        weight_decay_lambda: float
            权重衰减指数，作为控制L2正则化的超参数.

        Returns
        -------
        histories: list of dict
            保存训练集经过每一轮训练后的损失误差和监控指标值，如果训练时传入了验证集
            （即validation_data），则会同时保存有关验证集的损失误差和监控指标值.
        """
        if  self.optimizer_obj is None or \
                self.loss_func_obj is None or \
                self.metric_obj is None:
            err_msg = (
                "Before trainning the network, set up the optimizer, loss function "
                "and metric by using 'compile' method."
            )
            raise FitBeforeCompile(err_msg)

        self.optimizer_obj.lr = lr
        self.weight_decay_lambda = weight_decay_lambda

        self.build()
        histories = []
        for epoch in range(epochs):
            # 计时开始：
            start_time = time.time()
            # 训练网络：
            self.train(
                inputs=inputs,
                targets=targets,
                batch_size=batch_size,
                use_every_sample=use_every_sample
            )
            # 评估模型：
            result_train = self.evaluate(inputs, targets)
            # 验证模型：
            result_val = None
            if validation_data is not None:
                validate_data_type(tuple, "validation_data", validation_data)
                x_val, y_val = validation_data
                result_val = self.evaluate(x_val, y_val)
            # 计时结束：
            end_time = time.time()
            time_used = end_time - start_time
            # 收集历史数据：
            history = self._collect_historical_data(
                histories=histories,
                epoch=epoch,
                result_train=result_train,
                result_val=result_val,
                metric=self._metric_name,
                time_used=time_used
            )
            # 打印历史数据到终端：
            self._print_historical_data(
                epochs=epochs,
                verbose=verbose,
                history=history,
                metric=self._metric_name
            )
        return histories

    def loss(self, inputs, targets):
        """ 求解损失误差.

        Parameters
        ----------
        inputs: numpy.ndarray
            特征张量.
        targets: numpy.ndarray
            目标张量.

        Returns
        -------
        loss_value: float
            包含L2正则化惩罚项的损失误差.
        """

        outputs = self.predict(inputs)
        weights_decay = 0.0
        for layer in self:
            if hasattr(layer, "weights"):
                weights_decay += self.weight_decay_lambda * np.sum(layer.weights ** 2)
        weights_decay /= inputs.shape[0]
        loss_value = self.loss_func_obj(outputs, targets) + weights_decay
        return loss_value

    def predict(self, inputs):
        """ 使用神经网络进行预测.

        Parameters
        ----------
        inputs: numpy.ndarray
            特征张量.

        Returns
        -------
        outputs: numpy.ndarray
            预测值张量.
        """
        validate_data_type(np.ndarray, "inputs", inputs)

        outputs = inputs
        for layer in self:
            outputs = layer.forward(outputs)
        return outputs

    def train(self, inputs, targets, batch_size, use_every_sample=False):
        """ 执行网络的正向传播、反向传播以及使用优化器来更新每一层的参数.

        Parameters
        ----------
        inputs: numpy.ndarray
            训练集的特征张量.
        targets: numpy.ndarray
            训练集的目标张量，即分类任务中的标签，回归任务中的目标值.
        batch_size: int
            对训练集进行分批时，每一个批次所包含的样本数.
        use_every_sample: boolean
            在随机梯度下降算法中，因为用来训练的样本是随机挑选的，所以在同一轮训练里某些样本
            可能会被训练多次，而有些样本则没有得到训练. 而为了优化算法每一轮都可以遍历所有的、
            样本，这里提供了一种解决办法：在每一轮训练当中，先将训练集洗牌打乱，然后一个批次
            接一个批次地训练.
            因此，当use_every_sample设置为False时，可理解为使用与随机梯度下降相同的训练策略；
            当use_every_sample设置为True时，训练时将会遍历所有的样本.
        """
        if inputs.shape[0] == batch_size:
            sample_indices = np.arange(inputs.shape[0])
            np.random.shuffle(sample_indices)
            inputs = inputs[sample_indices]
            targets = targets[sample_indices]

        for i in self._batching(
            sample_num=inputs.shape[0],
            batch_size=batch_size,
            use_every_sample=use_every_sample,
        ):
            start = i * batch_size
            stop = (i + 1) * batch_size
            x = inputs[start:stop]
            t = targets[start:stop]
            self.loss(x, t)
            self.differentiate()
            self.optimizer_obj.update()
