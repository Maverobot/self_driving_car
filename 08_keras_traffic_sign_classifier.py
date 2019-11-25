#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
import pickle
import pandas as pd
import random
import cv2
import requests
from PIL import Image

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Adam = keras.optimizers.Adam
to_categorical = keras.utils.to_categorical
Dropout = keras.layers.Dropout
Flatten = keras.layers.Flatten
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D

ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator

np.random.seed(0)


def deserialize_pickel_file():
    # https://bitbucket.org/jadslim/german-traffic-signs.git
    with open("german-traffic-signs/train.p", "rb") as f:
        train_data = pickle.load(f)
    with open("german-traffic-signs/valid.p", "rb") as f:
        val_data = pickle.load(f)
    with open("german-traffic-signs/test.p", "rb") as f:
        test_data = pickle.load(f)
    return train_data, val_data, test_data


def checkValidation(X_train, y_train, X_val, y_val, X_test, y_test):
    assert (X_train.shape[0] == y_train.shape[0]
            ), "The number of images is not equal to the number of labels"
    assert (X_val.shape[0] == y_val.shape[0]
            ), "The number of images is not equal to the number of labels"
    assert (X_test.shape[0] == y_test.shape[0]
            ), "The number of images is not equal to the number of labels"
    assert X_train.shape[1:] == (
        32,
        32,
        3,
    ), "The dimensions of images are not 32 x 32 x 3"
    assert X_val.shape[1:] == (
        32,
        32,
        3,
    ), "The dimensions of images are not 32 x 32 x 3"
    assert X_test.shape[1:] == (
        32,
        32,
        3,
    ), "The dimensions of images are not 32 x 32 x 3"


def show_sample_images(data, X_train, y_train, num_samples=5):
    num_of_samples = []
    num_classes = len(set(y_train))
    fig, axe = plt.subplots(nrows=num_classes,
                            ncols=num_samples,
                            figsize=(5, 50))
    fig.tight_layout()
    for i, row in data.iterrows():
        x_selected = X_train[y_train == i]
        for j in range(num_samples):
            axe[i][j].imshow(
                x_selected[random.randint(0,
                                          len(x_selected) - 1)],
                cmap=plt.get_cmap("gray"),
            )
            axe[i][j].axis("off")
            if j == 2:
                axe[i][j].set_title(str(i) + "-" + row["SignName"])
                num_of_samples.append(len(x_selected))

    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribution of the training dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def equalize(img):
    return cv2.equalizeHist(img)


def preprocessing(img):
    return equalize(grayscale(img)) / 255


def leNet_model(num_classes):
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=num_classes, activation='softmax'))
    # Lower lr gives better accuracy
    model.compile(Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    # Import dataset
    train_data, val_data, test_data = deserialize_pickel_file()
    X_train, y_train = train_data["features"], train_data["labels"]
    X_val, y_val = val_data["features"], val_data["labels"]
    X_test, y_test = test_data["features"], test_data["labels"]

    num_classes = len(set(y_train))
    checkValidation(X_train, y_train, X_val, y_val, X_test, y_test)

    data = pd.read_csv("german-traffic-signs/signnames.csv")

    # show_sample_images(data, X_train, y_train)

    X_train = np.array(list(map(preprocessing, X_train)))
    X_val = np.array(list(map(preprocessing, X_val)))
    X_test = np.array(list(map(preprocessing, X_test)))

    # Add depth
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                              X_train.shape[2], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],
                            1)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)

    model = leNet_model(num_classes)

    try:
        model.load_weights("lenet_traffic_sign_weights.h5")
    except Exception as e:
        data_gen = ImageDataGenerator(width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      zoom_range=0.2,
                                      shear_range=0.1,
                                      rotation_range=10)
        history = model.fit_generator(data_gen.flow(X_train,
                                                    y_train,
                                                    batch_size=50),
                                      steps_per_epoch=2000,
                                      epochs=10,
                                      validation_data=(X_val, y_val),
                                      shuffle=True)
        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.legend(['training', 'validation'])
        # plt.title('accuracy')
        # plt.xlabel('epoch')
        model.save_weights("lenet_traffic_sign_weights.h5")

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test Score: ", score[0])
    print("Test Accuracy: ", score[1])

    url = 'https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg'
    r = requests.get(url, stream=True)
    img = Image.open(r.raw)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    #Preprocess image
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    #plt.imshow(img, cmap=plt.get_cmap('gray'))
    #Reshape reshape
    img = img.reshape(1, 32, 32, 1)
    #Test image
    sign_id = model.predict_classes(img)

    print("predicted sign: ", data['SignName'][sign_id])
    plt.show(block=True)


if __name__ == "__main__":
    main()
