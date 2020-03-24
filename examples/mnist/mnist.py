# -*- coding: utf-8 -*-

"""
本示例基于MNIST数据集实现手写数字识别.

鉴于MNIST数据集已经成为机器学习领域的‘hello world’，这里不再赘述关于该数据集的情况.
作为对比，使用keras的实验在同目录的mnist_keras.ipynb.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from fakeras.datasets.mnist import load_data
from fakeras.layer import Dense
from fakeras.network import NeuralNetwork


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    dtype = 'float32'
    x_train = x_train.reshape(60000, 784).astype(dtype) / 255
    y_train = one_hot_encode(y_train, 10, dtype)
    x_test = x_test.reshape(10000, 784).astype(dtype) / 255
    y_test = one_hot_encode(y_test, 10, dtype)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    model = NeuralNetwork()
    model.add(Dense(512, activation='relu', initialization='xavier', input_shape=(784,)))
    model.add(Dense(10, activation='softmax', initialization='xavier'))
    model.build()
    model.compile(optimizer='rmsprop', loss_func='cce', metric='acc')
    model.fit(inputs=x_train,
              targets=y_train,
              epochs=5,
              lr=0.01,
              batch_size=128,
              use_every_sample=True,
              verbose=True)
    result = model.evaluate(x_test, y_test)
    print(result)  #

    # 从测试集中随机选取10个样本做预测，然后与目标比较，其中包括将这10个样本还原为手写数字图片：
    indices = np.random.choice(len(x_test), size=10, replace=False)
    y_pred = model.predict(x_test[indices])
    print(np.argmax(y_pred, axis=1))
    print(np.argmax(y_test[indices], axis=1))
    x_to_pic = x_test[indices]
    x_to_pic = x_to_pic.reshape((10, 28, 28))
    for i, x in enumerate(x_to_pic):
        file = os.path.join('pictures', '%s.png' % i)
        plt.imsave(file, x)

def one_hot_encode(dataset, feature_num, dtype='float64'):
    assert isinstance(dataset, np.ndarray)
    sample_num = len(dataset)
    dataset_encoded = np.zeros((sample_num, feature_num), dtype=dtype)
    for sidx, tidx in enumerate(dataset):  # sample index and tag index
        dataset_encoded[sidx, tidx] = 1
    return dataset_encoded


if __name__ == '__main__':
    main()
