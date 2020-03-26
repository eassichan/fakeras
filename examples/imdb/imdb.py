# -*- coding: utf-8 -*-

"""
本示例基于IMDB数据集实现电影评论二分类任务，其中，数据集被划分为用于训练的25000条评论和用于测试的
25000条评论，训练集和测试集都包含50%的正面评论（1代表正面）和50%的负面评论（0代表负面）.

具体的实验内容参考了Francois Chollet撰写的《Python深度学习》的3.4小节，有关这部分的代码在同目录的
imdb_keras.ipynb.
"""

import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from fakeras.datasets.imdb import get_word_index, load_data
from fakeras.layer import Dense
from fakeras.network import NeuralNetwork


def main():
    (x_train, y_train), (x_test, y_test) = load_data(num_words=10000)
    x_test_src = copy.deepcopy(x_test)  # 用于还原电影评论文本数据
    dtype = 'float64'
    x_train = one_hot_encode(x_train, 10000, dtype)
    x_test = one_hot_encode(x_test, 10000, dtype)
    y_train = np.asarray(y_train).astype(dtype)
    y_train.resize(25000, 1)
    y_test = np.asarray(y_test).astype(dtype)
    y_test.resize(25000, 1)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    (x_train, y_train), (x_val, y_val) = validate(x_train, y_train, test_ratio=0.4)

    model = build_model()
    histories = model.fit(inputs=x_train,
                          targets=y_train,
                          epochs=20,
                          lr=0.001,
                          batch_size=512,
                          use_every_sample=True,
                          verbose=True,
                          validation_data=(x_val, y_val))
    plot_learning_curves(histories, metric_name='bin_acc')

    # 观察上述步骤得到的损失曲线和精度曲线，可以看出模型在第4~6轮之后开始过拟合，
    # 因此，从头开始训练新的网络，训练4轮，然后在测试数据上评估模型：
    del model  # 释放上一个网络
    model = build_model()
    model.fit(inputs=x_train,
              targets=y_train,
              epochs=4,
              lr=0.001,
              batch_size=512,
              use_every_sample=True,
              verbose=True)
    result = model.evaluate(x_test, y_test)
    print(result)  # 损失值和识别精度大概在0.33和86%左右

    # 先从测试集随机选取10个样本，然后将特征数据还原为电影评论，最后对比模型的预测值和真实值：
    # indices = np.random.choice(x_test.shape[1], size=10, replace=False)
    indices = np.random.choice(x_test.shape[0], size=10, replace=False)
    x_test = x_test[indices]
    reviews = index_to_word(x_test_src[indices])
    y_test = y_test[indices]
    y_pred = model.predict(x_test)

    r_rng = range(len(reviews))
    p_rng = range(y_pred.shape[0])
    t_rng = range(y_test.shape[0])
    review_pred_target = []
    for r, p, t in zip(r_rng, p_rng, t_rng):
        review_pred_target.append(dict(review=reviews[r],
                                       predict=float(y_pred[p, 0]),
                                       target=float(y_test[t, 0])))
    # for d in review_pred_target:

    #     for k, v in d.items():
    #         print(k, ': ', v)
    #     print('=' * 80)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'outputs'), exist_ok=True)
    file_name = os.path.join(os.path.dirname(__file__), 'outputs', 'reviews.json')
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(review_pred_target, f, ensure_ascii=False, indent=4)

def build_model():
    model = NeuralNetwork()
    model.add(Dense(16, 'relu', 'xavier', input_shape=(10000,)))
    model.add(Dense(16, 'relu', 'xavier'))
    model.add(Dense(1, 'sigmoid', 'xavier'))
    model.build()
    model.compile(optimizer='rmsprop', loss_func='bce', metric='bin_acc')
    return model

def one_hot_encode(dataset, feature_num, dtype='float64'):
    assert isinstance(dataset, np.ndarray)
    sample_num = len(dataset)
    dataset_encoded = np.zeros((sample_num, feature_num), dtype=dtype)
    for sidx, widx in enumerate(dataset):  # sample index and word index
        # dataset_encoded[widx, sidx] = 1
        dataset_encoded[sidx, widx] = 1
    return dataset_encoded

def validate(x, y, test_ratio=0.2, seed=42, shuffle=False):
    if shuffle:
        indices = np.arange(x.shape[0])
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        x = x[indices]
        y = y[indices]

    break_point = int(len(x) * (1 - test_ratio))
    x_train = np.array(x[:break_point])
    y_train = np.array(y[:break_point])
    x_test = np.array(x[break_point:])
    y_test = np.array(y[break_point:])
    return (x_train, y_train), (x_test, y_test)

def plot_learning_curves(histories, metric_name):
    """ 绘制损失曲线和监控指标曲线. """
    os.makedirs(os.path.join(os.path.dirname(__file__), 'pictures'), exist_ok=True)
    dir_name = os.path.join(os.path.dirname(__file__), 'pictures')
    os.makedirs(dir_name, exist_ok=True)
    Epochs = [histories[i]['epoch'] for i in range(len(histories))]
    loss = [histories[i]['loss'] for i in range(len(histories))]
    loss_val = [histories[i]['loss_val'] for i in range(len(histories))]
    metric = [histories[i][metric_name] for i in range(len(histories))]
    metric_val = [histories[i][metric_name+'_val'] for i in range(len(histories))]

    plot_loss_curves(Epochs, loss, loss_val, dir_name)
    plot_metric_curves(Epochs, metric, metric_val, dir_name)

def plot_loss_curves(epochs, loss, loss_val, dir_name):
    """ 绘制损失曲线. """
    file_name = os.path.join(dir_name, 'loss.png')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(epochs, loss, 'bo', label="Training Loss")
    plt.plot(epochs, loss_val, 'b', label="Validation Loss")
    plt.legend()
    plt.savefig(file_name)
    plt.close()

def plot_metric_curves(epochs, metric, metric_val, dir_name):
    """ 绘制监控指标曲线. """
    file_name = os.path.join(dir_name, 'metric.png')
    plt.title("Training and validation Metric")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.plot(epochs, metric, 'bo', label="Training Metric")
    plt.plot(epochs, metric_val, 'b', label="Validation Metric")
    plt.legend()
    plt.savefig(file_name)
    plt.close()

def index_to_word(indices):
    word_index = get_word_index()
    index_word = {i: w for w, i in word_index.items()}
    docs = []
    for idx in indices:
        doc = ' '.join([index_word.get(i, '?') for i in idx])
        docs.append(doc)
    return docs


if __name__ == '__main__':
    main()
