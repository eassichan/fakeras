# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np

from fakeras import NeuralNetwork
from fakeras.datasets import mnist
from fakeras.layer import Conv2D, Dense, Flatten, MaxPool2D


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    # x_train = x_train[:1000]
    # y_train = y_train[:1000]
    model = build_model()
    histories = model.fit(
        inputs=x_train,
        targets=y_train,
        batch_size=64,
        epochs=5,
        lr=0.001,
        use_every_sample=True,
        verbose=True
    )
    result = model.evaluate(x_test, y_test)
    print(result)

    # 从测试集中随机选取10个样本做预测，然后与目标比较，其中包括将这10个样本还原为手写数字图片：
    indices = np.random.choice(len(x_test), size=10, replace=False)
    # y_pred = model.predict(x_test[:, indices])
    y_pred = model.predict(x_test[indices])
    print(np.argmax(y_pred, axis=1))
    print(np.argmax(y_test[indices], axis=1))
    x_to_pic = x_test[indices]
    x_to_pic = x_to_pic.reshape((10, 28, 28))
    os.makedirs('pictures', exist_ok=True)
    for i, x in enumerate(x_to_pic):
        file = os.path.join('pictures', '%s.png' % i)
        plt.imsave(file, x)
    return

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    dtype = 'float32'
    x_train = x_train.reshape(60000, 1, 28, 28).astype(dtype) / 255
    x_test = x_test.reshape(10000, 1, 28, 28).astype(dtype) / 255

    y_train = one_hot_encode(y_train, feature_num=10, dtype=dtype)
    y_test = one_hot_encode(y_test, feature_num=10, dtype=dtype)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print('-' * 80)
    return (x_train, y_train), (x_test, y_test)

def one_hot_encode(dataset, feature_num, dtype='float64'):
    assert isinstance(dataset, np.ndarray)
    sample_num = len(dataset)
    dataset_encoded = np.zeros((sample_num, feature_num), dtype=dtype)
    for sidx, widx in enumerate(dataset):  # sample index and word index
        dataset_encoded[sidx, widx] = 1
    return dataset_encoded

def build_model():
    model = NeuralNetwork()
    model.add(Conv2D(32, (3, 3), activation='relu', initialization='xavier',
                     input_shape=(1, 28, 28)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', initialization='xavier'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', initialization='xavier'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', initialization='xavier'))
    model.add(Dense(10, activation='softmax', initialization='xavier'))

    # model.build()
    model.compile(optimizer='rmsprop', loss_func='cce', metric='acc')
    return model


if __name__ == '__main__':
    main()
