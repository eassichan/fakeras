# -*- coding: utf-8 -*-

import os

import numpy as np


def load_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "data", "mnist.npz")
    else:
        assert os.path.exists(path), "Path '%s' not found." % path

    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
