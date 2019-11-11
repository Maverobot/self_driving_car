#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow import keras

to_categorical = keras.utils.to_categorical


def main():
    n_pts = 5
    centers = [[-1, 1], [-1, -1], [1, -1]]
    X, y = datasets.make_blobs(
        n_samples=n_pts, centers=centers, random_state=123, cluster_std=0.4
    )
    print(X)
    print(y)
    y_cat = to_categorical(y)
    print(y_cat)

    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    plt.scatter(X[y == 2, 0], X[y == 2, 1])
    plt.show(block=True)


if __name__ == "__main__":
    main()
