#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow import keras


def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
    y_span = np.linspace(min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred = model.predict(grid)
    z = pred.reshape(xx.shape)
    plt.contourf(xx, yy, z)


def main():
    np.random.seed(0)
    n_pts = 500
    X, y = datasets.make_circles(n_samples=n_pts, random_state=0, noise=0.1, factor=0.2)

    model = keras.Sequential()
    model.add(keras.layers.Dense(units=4, input_shape=(2,), activation="sigmoid"))
    model.add(keras.layers.Dense(units=1, activation="sigmoid"))
    model.compile(
        keras.optimizers.Adam(lr=0.01), loss="binary_crossentropy", metrics=["accuracy"]
    )
    h = model.fit(x=X, y=y, verbose=1, batch_size=20, epochs=100, shuffle=True)

    plt.figure(0)
    plot_decision_boundary(X, y, model)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="b")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="g")
    plt.title("decision boundary")

    x = 0.1
    y = 0.1
    point = np.array([[x, y]])
    prediction = model.predict(point)
    plt.plot([x], [y], marker="o", markersize=10, color="red")
    print("Prediction for point " + str(point) + " is :" + str(prediction))

    plt.figure(1)
    plt.plot(h.history["acc"])
    plt.plot(h.history["loss"])
    plt.xlabel("epoch")
    plt.legend(["accuracy", "loss"])
    plt.title("accuracy and loss")

    plt.show(block=True)


if __name__ == "__main__":
    main()
