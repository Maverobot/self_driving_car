#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Adam = keras.optimizers.Adam

np.random.seed(0)


def load_model():
    model = Sequential()
    model.add(Dense(units=50, input_dim=1, activation='sigmoid'))
    model.add(Dense(units=30, activation='sigmoid'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=0.01), loss='mse')
    return model


def main():
    points = 5000
    x = np.linspace(-3, 3, points)
    y = np.sin(x) + np.random.uniform(-0.5, 0.5, points)
    plt.scatter(x, y)

    model = load_model()
    model.fit(x,
              y,
              batch_size=100,
              shuffle=True,
              verbose=1,
              validation_split=0.2,
              epochs=50)

    predictions = model.predict(x)
    plt.plot(x, predictions, 'ro')
    plt.show(block=True)


if __name__ == '__main__':
    main()
