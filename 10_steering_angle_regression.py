#!/usr/bin/env python

import os
import random

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras

Conv2D = keras.layers.Conv2D
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Dropout = keras.layers.Dropout
Adam = keras.optimizers.Adam


def load_data():
    columns = [
        'center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'
    ]
    data_dir = '../simulator/data'
    data = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'),
                       names=columns)
    pd.set_option('display.max_colwidth', -1)
    data['center'] = data['center'].apply(path_leaf)
    data['left'] = data['left'].apply(path_leaf)
    data['right'] = data['right'].apply(path_leaf)
    return data


def data_balancing(data, num_bins=25, plot=False):
    hist, bins = np.histogram(data['steering'], num_bins)
    # center bins around zero
    centered_bin = (bins[:-1] + bins[1:]) * 0.5

    max_samples_per_bin = 200

    remove_list = []
    for bin_idx in range(num_bins):
        list_ = []
        for data_idx in range(len(data['steering'])):
            if data['steering'][data_idx] >= bins[bin_idx] and data[
                    'steering'][data_idx] <= bins[bin_idx + 1]:
                list_.append(data_idx)
        list_ = shuffle(list_)
        remove_list.extend(list_[max_samples_per_bin:])
    data.drop(data.index[remove_list], inplace=True)

    hist, bins = np.histogram(data['steering'], num_bins)
    if plot:
        plt.bar(centered_bin, hist, width=0.03)
        plt.plot((np.min(data['steering']), np.max(data['steering'])),
                 (max_samples_per_bin, max_samples_per_bin))
    return data


def load_img_steering(data):
    data_dir = '../simulator/data/IMG'
    image_paths = []
    steerings = []
    for data_idx in range(len(data)):
        indexed_data = data.iloc[data_idx]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_paths.append(os.path.join(data_dir, center.strip()))
        steerings.append(float(indexed_data[3]))
    image_paths = np.asarray(image_paths)
    steerings = np.asarray(steerings)
    return image_paths, steerings


def path_leaf(path):
    return os.path.basename(path)


def img_preprocess(img_path):
    img = mpimg.imread(img_path)
    # Crop only the relavant image region
    img = img[60:135, :, :]
    # Use YUV color space as nvidia NN model recommended
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Use GaussianBlur to suppress noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Resize image to a fixed size
    img = cv2.resize(img, (200, 66))
    # Normalization
    img = img / 255
    return img


def random_test(image_paths):
    image_path = image_paths[100]
    original_img = mpimg.imread(image_path)
    preprocessed_img = img_preprocess(image_path)
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
    axes[0].imshow(original_img)
    axes[0].set_title("original image")
    axes[1].imshow(preprocessed_img)
    axes[1].set_title("preprocessed image")


def nvidia_model():
    model = Sequential()
    model.add(
        Conv2D(input_shape=(66, 200, 3),
               filters=24,
               kernel_size=(5, 5),
               strides=(2, 2),
               activation='elu'))
    model.add(
        Conv2D(filters=36,
               kernel_size=(5, 5),
               strides=(2, 2),
               activation='elu'))
    model.add(
        Conv2D(filters=48,
               kernel_size=(5, 5),
               strides=(2, 2),
               activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(units=100, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=50, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))
    model.compile(Adam(lr=0.001), loss='mse')
    return model


def main():
    num_bins = 25
    data = load_data()
    data = data_balancing(data, num_bins)
    image_paths, steerings = load_img_steering(data)

    X_train, X_valid, y_train, y_valid = train_test_split(image_paths,
                                                          steerings,
                                                          test_size=0.2,
                                                          random_state=0)
    # print("Training samples: {}\nValid Samples: {}".format(
    #     len(X_train), len(X_valid)))
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
    # axes[0].set_title("Training set")
    # axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
    # axes[1].set_title("Validation set")

    X_train = np.array(list(map(img_preprocess, X_train)))
    X_valid = np.array(list(map(img_preprocess, X_valid)))

    # plt.imshow(X_train[random.randint(0, len(X_train) - 1)])
    # plt.axis("off")
    # print(X_train.shape)
    model = nvidia_model()
    print(model.summary())

    try:
        model.load_weights("nvidia_net_weights.h5")
    except Exception:
        history = model.fit(X_train,
                            y_train,
                            epochs=35,
                            validation_data=(X_valid, y_valid),
                            verbose=1,
                            batch_size=100,
                            shuffle=True)
        model.save_weights("nvidia_net_weights.h5")
        model.save("nvidia_net_model.h5")

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['training', 'validation'])
        plt.title('Loss')
        plt.xlabel('Epoch')


if __name__ == '__main__':
    main()
    plt.show(True)
