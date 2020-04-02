# -*- coding: utf-8 -*-

"""
本示例基于路透社数据集实现新闻主题多分类.

路透社数据集包含46个新闻主题，总共有8982个样本用于训练，2246个样本用于测试.
具体的实验内容参考了Francois Chollet撰写的《Python深度学习》的3.5小节，有关这部分的代码
在同目录的reuters.ipynb.
"""

import copy
import os
import json

import matplotlib.pyplot as plt
import numpy as np

from fakeras.datasets.reuters import load_data, get_word_index
from fakeras.layer import Dense
from fakeras.network import NeuralNetwork


def main():
    (x_train, y_train), (x_test, y_test) = load_data(num_words=10000)
    x_test_src = copy.deepcopy(x_test)
    dtype = 'float64'
    x_train = one_hot_encode(x_train, 10000, dtype=dtype)
    x_test = one_hot_encode(x_test, 10000, dtype=dtype)
    y_train = one_hot_encode(y_train, 46, dtype=dtype)
    y_test = one_hot_encode(y_test, 46, dtype=dtype)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    (x_train, y_train), (x_val, y_val) = validate(x_train, y_train, num_val_sample=1000)

    model = build_model()
    histories = model.fit(inputs=x_train,
                          targets=y_train,
                          epochs=20,
                          lr=0.001,
                          batch_size=512,
                          use_every_sample=True,
                          verbose=True,
                          validation_data=(x_val, y_val))
    # result = model.evaluate(x_test, y_test)
    # print(result)
    plot_learning_curves(histories, metric_name='acc')

    # 观察上述步骤得到的损失曲线和精度曲线，可以看出模型在第9~10轮左右开始过拟合，
    # 因此，从头开始训练新的网络，训练10轮，然后在测试数据上评估模型：
    del model  # 释放上一个网络占用的内存
    model = build_model()
    model.fit(
        inputs=x_train,
        targets=y_train,
        epochs=10,
        lr=0.001,
        batch_size=512,
        use_every_sample=True,
        verbose=True
    )
    result = model.evaluate(x_test, y_test)
    print(result)  # 损失值和识别精度大概在1.0和77%左右

    # 先从测试集随机选取10个样本，然后将特征数据还原为新闻文本，最后对比模型的预测值和真实值：
    indices = np.random.choice(x_test.shape[0], size=10, replace=False)
    # x_test = x_test[:, indices]
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
                                       predict=float(np.argmax(y_pred[p, :])),
                                       target=float(np.argmax(y_test[t, :])),
                                       categoriy=get_categories(np.argmax(y_test[t, :]))))
    # for d in review_pred_target:
    #     for k, v in d.items():
    #         print(k, ': ', v)
    #     print('=' * 80)
    # os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), 'outputs')))
    # file_name = os.path.join(os.path.dirname(__file__), 'outputs', 'news.json')
    os.makedirs('outputs', exist_ok=True)
    file_name = os.path.join('outputs', 'news.json')
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(review_pred_target, f, ensure_ascii=False, indent=4)


def one_hot_encode(docs, dim, dtype):
    docs_encoded = np.zeros((len(docs), dim), dtype=dtype)
    for i, doc in enumerate(docs):
        docs_encoded[i, doc] = 1
    return docs_encoded


def build_model():
    model = NeuralNetwork()
    model.add(Dense(64, 'relu', 'xavier', input_shape=(10000,)))
    model.add(Dense(64, 'relu', 'xavier'))
    model.add(Dense(46, 'softmax', 'xavier'))
    model.build()
    model.compile(optimizer='rmsprop', loss_func='cce', metric='acc')
    return model


def validate(x, y, num_val_sample):
    x_train = x[num_val_sample:]
    y_train = y[num_val_sample:]
    x_val = x[:num_val_sample]
    y_val = y[:num_val_sample]
    return (x_train, y_train), (x_val, y_val)


def plot_learning_curves(histories, metric_name):
    """ 绘制损失曲线和监控指标曲线. """
    dir_name = os.path.join(os.path.dirname(__file__), 'pictures')
    os.makedirs(dir_name, exist_ok=True)
    Epochs = [histories[i]['epoch'] for i in range(len(histories))]
    loss = [histories[i]['loss'] for i in range(len(histories))]
    loss_val = [histories[i]['loss_val'] for i in range(len(histories))]
    metric = [histories[i][metric_name] for i in range(len(histories))]
    metric_val = [histories[i][metric_name+'_val'] for i in range(len(histories))]

    plot_loss_curves(Epochs, loss, loss_val, dir_name)
    plot_metric_curves(Epochs, metric, metric_val, dir_name)


def plot_loss_curves(Epochs, loss, loss_val, dir_name):
    """ 绘制损失曲线. """
    file_name = os.path.join(dir_name, 'loss.png')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(Epochs, loss, 'bo', label="Training Loss")
    plt.plot(Epochs, loss_val, 'b', label="Validation Loss")
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


def get_categories(cate_num):
    # 46个主题标签如下：
    # 数据来源：https://github.com/keras-team/keras/issues/12072
    categories = ['cocoa', 'grain', 'veg-oil', 'earn', 'acq', 'wheat', 'copper', 'housing',
                  'money-supply', 'coffee', 'sugar', 'trade', 'reserves', 'ship', 'cotton',
                  'carcass', 'crude', 'nat-gas', 'cpi', 'money-fx', 'interest', 'gnp',
                  'meal-feed', 'alum', 'oilseed', 'gold', 'tin', 'strategic-metal', 'livestock',
                  'retail', 'ipi', 'iron-steel', 'rubber', 'heat', 'jobs', 'lei', 'bop', 'zinc',
                  'orange', 'pet-chem', 'dlr', 'gas', 'silver', 'wpi', 'hog', 'lead']
    assert cate_num in range(-len(categories), len(categories))
    return categories[cate_num]


if __name__ == '__main__':
    main()