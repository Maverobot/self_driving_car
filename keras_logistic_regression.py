#!/usr/bin/env python
"""
This is an example to use Keras to solve logistic regression problem.
"""

import numpy as np
import matplotlib.pyplot as plt

# Tested with tensorflow 1.9
import tensorflow as tf
from tensorflow import keras


def main():
    """
    This is the main workflow of logistic regression.
    """
    n_pts = 1000
    np.random.seed(0)
    Xa = np.array([np.random.normal(13, 2, n_pts), np.random.normal(12, 2, n_pts)]).T
    Xb = np.array([np.random.normal(8, 2, n_pts), np.random.normal(6, 2, n_pts)]).T

    X = np.vstack((Xa, Xb))
    y = np.matrix([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts * 2, 1)

    # plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
    # plt.scatter(X[n_pts:, 0], X[n_pts:, 1])

    model = keras.Sequential()
    model.add(keras.layers.Dense(units=1, input_shape=(2,), activation="sigmoid"))

    # Adam automatically finds the learning rate.
    adam = keras.optimizers.Adam(lr=0.1)
    model.compile(adam, loss="binary_crossentropy", metrics=["accuracy"])
    h = model.fit(x=X, y=y, verbose=1, batch_size=100, epochs=200, shuffle="true")

    plt.plot(h.history["loss"])
    plt.title("loss")
    plt.xlabel("epoch")
    plt.legend(["loss"])
    plt.show(block=True)


if __name__ == "__main__":
    main()
