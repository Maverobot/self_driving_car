#!/usr/bin/env python3
# pylint: disable=C,locally-disabled

import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import random
import h5py

mnist = keras.datasets.mnist
Sequential = keras.Sequential
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Conv2D = keras.layers.Conv2D
Dropout = keras.layers.Dropout
MaxPooling2D = keras.layers.MaxPool2D
Adam = keras.optimizers.Adam
to_categorical = keras.utils.to_categorical


def check_dataset(X_train, y_train, X_test, y_test):
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    assert X_train.shape[1:] == (28, 28), "The dimensions of the images are not 28x28"
    assert X_test.shape[1:] == (28, 28), "The dimensions of the images are not 28x28"


def show_sample_images(X_train, y_train, num_samples=5):
    num_of_samples = []
    num_classes = len(set(y_train))
    fig, axe = plt.subplots(nrows=num_classes, ncols=num_samples, figsize=(10, 10))
    fig.tight_layout()
    for i in range(num_classes):
        x_selected = X_train[y_train == i]
        for j in range(num_samples):
            axe[i][j].imshow(
                x_selected[random.randint(0, len(x_selected) - 1)],
                cmap=plt.get_cmap("gray"),
            )
            axe[i][j].axis("off")
            if j == 2:
                axe[i][j].set_title(str(i))
                num_of_samples.append(len(x_selected))

    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribution of the training dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")


def create_lenet_model(input_shape, num_classes):
    model = Sequential()
    model.add(
        Conv2D(
            30,
            (5, 5),
            input_shape=input_shape,
            activation="relu",
            strides=1,
            padding="valid",
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(
            15,
            (3, 3),
            input_shape=input_shape,
            activation="relu",
            strides=1,
            padding="valid",
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(units=500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))
    model.compile(Adam(lr=0.01), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main(url):
    np.random.seed(0)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train.shape)
    print(y_train.shape)
    check_dataset(X_train, y_train, X_test, y_test)
    # show_sample_images(X_train, y_train)
    num_classes = len(set(y_train))

    # Use one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Normalize pixel intensity.
    # Consider the sigmoid function where e^255 would be 5.5E110 while e^1 is just 2.7.
    # It would be much easier to train after the normalization.
    X_train = X_train / 255
    X_test = X_test / 255

    # Flatten the data
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    model = create_lenet_model(X_train.shape[1:4], num_classes)
    print(model.summary())
    try:
        model.load_weights("mnist_weights.h5")
    except Exception as e:
        h = model.fit(
            x=X_train,
            y=y_train,
            validation_split=0.1,
            verbose=1,
            epochs=10,
            batch_size=200,
            shuffle=True,
        )
        plt.figure(1)
        plt.plot(h.history["loss"])
        plt.plot(h.history["val_loss"])
        plt.legend(["loss", "val_loss"])
        plt.title("loss")
        plt.xlabel("epoch")

    model.save_weights("mnist_weights.h5")

    score = model.evaluate(X_test, y_test, verbose=0)
    print(type(score))
    print("Test score: ", score[0])
    print("Test accuracy: ", score[1])

    # Test with online image
    import requests
    from PIL import Image

    response = requests.get(url, stream=True)
    print(response)
    img = Image.open(response.raw)
    import cv2

    img_array = np.asarray(img)
    img_resized = cv2.resize(img_array, (28, 28))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.bitwise_not(img_gray)
    plt.figure(2)
    plt.imshow(img_gray, cmap=plt.get_cmap("gray"))

    img_gray = img_gray / 255
    img_gray = img_gray.reshape(1, img_gray.shape[0], img_gray.shape[1], 1)
    label = model.predict_classes(img_gray)
    print("predicted digit: ", str(label))

    plt.show(block=True)
    return True


if __name__ == "__main__":
    url = sys.argv[1]
    main(url)
