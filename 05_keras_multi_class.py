#!/usr/bin/env python
# pylint: disable=C,locally-disabled

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow import keras

to_categorical = keras.utils.to_categorical


def plot_decision_boundary(X, model):
    x_span = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1)
    y_span = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred = model.predict_classes(grid)
    z = pred.reshape(xx.shape)
    plt.contourf(xx, yy, z)


def main():
    np.random.seed(0)
    n_pts = 500
    centers = [[-1, 1], [-1, -1], [1, -1], [1, 1], [0, 0]]
    X, y = datasets.make_blobs(
        n_samples=n_pts, centers=centers, random_state=123, cluster_std=0.4
    )
    y_cat = to_categorical(y)

    model = keras.Sequential()
    model.add(keras.layers.Dense(units=5, input_shape=(2,), activation="softmax"))
    model.compile(
        keras.optimizers.Adam(lr=0.1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(x=X, y=y_cat, verbose=1, batch_size=20, epochs=50)
    plot_decision_boundary(X, model)

    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    plt.scatter(X[y == 2, 0], X[y == 2, 1])
    plt.scatter(X[y == 3, 0], X[y == 3, 1])
    plt.scatter(X[y == 4, 0], X[y == 4, 1])

    x = 0.5
    y = 0.5
    point = np.array([[x, y]])
    prediction = model.predict_classes(point)
    plt.plot([x], [y], marker="o", markersize=10, color="r")
    print("prediction is ", prediction)

    plt.show(block=True)


if __name__ == "__main__":
    main()
