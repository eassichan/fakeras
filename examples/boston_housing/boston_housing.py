# -*- coding: utf-8 -*-

"""
本示例基于波士顿房产数据集实现了多变量线性回归.

在这个例子当中，主要的任务就是预测20世纪70年代中期波士顿郊区房屋价格的中位数，使用的数据集包含404个
训练样本，102个测试样本，其中每个样本都含有13个特征，且每个特征都有不同的取值范围，每个样本的目标值
代表着该区域的房价中位数.
具体的实验内容参考了Francois Chollet撰写的《Python深度学习》的3.6小节，有关这部分的代码在同目录的
boston_housing_keras.ipynb.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np

from fakeras.datasets.boston_housing import load_data
from fakeras.layer import Dense
from fakeras.network import NeuralNetwork


def main():
    (x_train, y_train), (x_test, y_test) = load_data_set()
    # 预训练，先训练出一个过拟合的模型，并将训练指标绘制成曲线图，
    # 然后观察曲线图，记下模型开始过拟合的轮数，
    # 最后按照刚才记下的轮数来重新训练模型：
    pre_train(x_train, y_train, batch_size=1, num_fold=4, num_epoch=500)
    # 观察曲线可知模型大概在80轮左右开始过拟合，所以重新训练模型80遍：
    train(x_train, y_train, x_test, y_test, batch_size=1, num_epoch=50)  # 最终的MAE在2.2~2.5之间

def load_data_set():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, x_test = standardize(x_train, x_test)
    y_train.resize(404, 1)
    y_test.resize(102, 1)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    return (x_train, y_train), (x_test, y_test)

def standardize(x_train, x_test, axis=0, keepdims=False):
    """ 将数据标准化.

        对输入数据的每一个特征，减去特征平均值，再除以标准差，这样得到的特征平均值为0，
        标准差为1.
    """
    mean = x_train.mean(axis=axis, keepdims=keepdims)
    x_train -= mean
    std = x_train.std(axis=axis, keepdims=keepdims)
    x_train /= std

    # 注意，用于测试数据标准化的均值和标准差都是在训练数据上计算得到的：
    x_test -= mean
    x_test /= std
    return x_train, x_test

def build_model():
    model = NeuralNetwork()
    model.add(Dense(64, 'relu', 'xavier', input_shape=(13,)))
    model.add(Dense(64, 'relu', 'xavier'))
    model.add(Dense(1, 'identity', 'xavier'))
    model.build()
    model.compile(optimizer='rmsprop', loss_func='mse', metric='mae')
    return model

def pre_train(x_train, y_train, batch_size, num_fold, num_epoch):
    hists = k_ford_validate(x_train, y_train, batch_size, num_fold, num_epoch)
    plot_learning_curves(hists)

def k_ford_validate(x_train, y_train, batch_size, num_fold, num_epoch, lr=0.001):
    """ 使用K折验证训练模型. """
    num_sample = x_train.shape[0]
    fold_size = math.ceil(num_sample / num_fold)  # 上取整

    all_histories = []
    for k in range(num_fold):
        print("Processing fold #", k)

        start = k * fold_size
        stop = (k + 1) * fold_size
        x_val = x_train[start:stop]
        y_val = y_train[start:stop]
        x_train_part = np.concatenate([x_train[:start],
                                       x_train[stop:]],
                                      axis=0)
        y_train_part = np.concatenate([y_train[:start],
                                       y_train[stop:]],
                                      axis=0)

        model = build_model()
        histories = model.fit(inputs=x_train_part,
                              targets=y_train_part,
                              batch_size=batch_size,
                              epochs=num_epoch,
                              lr=lr,
                              validation_data=(x_val, y_val),
                              use_every_sample=True,
                              verbose=True)
        # all_histories是一个二维列表，第一维代表折数，
        # 第二维代表轮数，最里层的元素是包含mse和mae的字典
        all_histories.append(histories)
        del model

    # 对每一折的历史数据汇总求均值，然后按轮数收集：
    hist_index_by_epoch = []
    for e in range(num_epoch):
        mse = 0.
        mae = 0.
        mse_val = 0.
        mae_val = 0.
        time_used = 0.
        for k in range(num_fold):
            hist = all_histories[k][e]  # type(hist) == dict
            mse += hist['loss']
            mae += hist['mae']
            mse_val += hist['loss_val']
            mae_val += hist['mae_val']
            time_used += hist['time_used']
        mse /= num_fold
        mae /= num_fold
        mse_val /= num_fold
        mae_val /= num_fold
        hist_index_by_epoch.append(dict(mse=mse, mae=mae, mse_val=mse_val, mae_val=mae_val,
                                        time_used=time_used, epoch=e))
    return hist_index_by_epoch

def smooth(x, beta=0.9):
    """ 使用指数加权平均让曲线更平滑. """
    y = []
    for i in x:
        # if y:
        if len(y) > 0:
            pre = y[-1]
            y.append(beta * pre + (1 - beta) * i)
        else:
            y.append(i)
    return y

def plot_learning_curves(histories):
    """ 绘制监控指标曲线. """
    dir_name = os.path.join(os.path.dirname(__file__), 'pictures')
    os.makedirs(dir_name, exist_ok=True)
    # 删除前10个样本点，因为它们的取值范围较其他样本大；
    # 将每个样本点替换为前面的样本点的指数加权平均值，以得到平滑的曲线：
    epochs = [histories[i]['epoch'] for i in range(len(histories))][10:]
    loss_val = smooth([histories[i]['mse_val'] for i in range(len(histories))][10:])
    metric_val = smooth([histories[i]['mae_val'] for i in range(len(histories))][10:])

    plot_loss_curves(epochs, loss_val, dir_name)
    plot_metric_curves(epochs, metric_val, dir_name)

def plot_loss_curves(epochs, loss_val, dir_name):
    """ 绘制损失曲线. """
    file_name = os.path.join(dir_name, 'loss.png')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(epochs, loss_val, 'b', label="Validation Loss")
    plt.legend()
    plt.savefig(file_name)
    plt.close()

def plot_metric_curves(epochs, metric_val, dir_name):
    """ 绘制监控指标曲线. """
    file_name = os.path.join(dir_name, 'metric.png')
    plt.title("Training and validation Metric")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.plot(epochs, metric_val, 'b', label="Validation Metric")
    plt.legend()
    plt.savefig(file_name)
    plt.close()

def train(x_train, y_train, x_test, y_test, batch_size, num_epoch, lr=0.001):
    model = build_model()
    model.fit(inputs=x_train,
              targets=y_train,
              batch_size=batch_size,
              epochs=num_epoch,
              lr=lr,
              use_every_sample=True,
              verbose=True)
    result = model.evaluate(x_test, y_test)
    print(result)


if __name__ == '__main__':
    main()
