# -*- coding: utf-8 -*-

import os

import numpy as np


def load_data(path=None, test_ratio=0.2, seed=42):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "data", "boston_housing.npz")
    else:
        assert os.path.exists(path), "Path '%s' not found." % path
    assert 0 < test_ratio < 1, "The ratio of testing set is wrong."

    with np.load(path, allow_pickle=True) as f:
        x = f["x"]
        y = f["y"]
    indices = np.arange(len(x))
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
