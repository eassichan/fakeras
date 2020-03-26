# -*- coding: utf-8 -*-

import json
import os

import numpy as np


def load_data(path=None, num_words=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "data", "imdb.npz")
    else:
        assert os.path.exists(path), "Path '%s' not found." % path

    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    train_indices = np.arange(len(x_train))
    x_train, y_train = _shuffle(train_indices, x_train, y_train)
    test_indices = np.arange(len(x_test))
    x_test, y_test = _shuffle(test_indices, x_test, y_test)
    xs = np.concatenate([x_train, x_test])
    ys = np.concatenate([y_train, y_test])

    if num_words is None:
        num_words = max(max([i for i in xs]))
    xs = [[w for w in x if 0 <= w < num_words] for x in xs]

    break_point = len(x_train)
    x_train, y_train = np.array(xs[:break_point]), np.array(ys[:break_point])
    x_test, y_test = np.array(xs[break_point:]), np.array(ys[break_point:])
    return (x_train, y_train), (x_test, y_test)


def _shuffle(indices, x, y):
    # indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    return x, y


def get_word_index(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "data", "imdb_word_index.json")
    else:
        assert os.path.exists(path), "Path '%s' not found." % path

    with open(path) as f:
        word_index = json.load(f)
    return word_index


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # 将整数索引序列解码成单词序列：
    word_index = get_word_index()  # key is word, value is numeric index
    index_word = {i: w for w, i in word_index.items()}
    decoded_review = " ".join(index_word.get(i, "?") for i in x_train[0])
    print(decoded_review)
