#!/usr/bin/env python

import base64
from io import BytesIO

import cv2
# conda install -c conda-forge eventlet
import eventlet
import numpy as np
# conda install -c conda-forge python-socketio
import socketio
# conda install -c anaconda flask
from flask import Flask
# conda install -c anaconda pillow
from PIL import Image
from tensorflow import keras

sio = socketio.Server()

app = Flask(__name__)


def img_preprocess(img):
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


speed_limit = 20


# Data back from simulation
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed / speed_limit
    # print("{} {} {}".format(steering_angle, throttle, speed_limit))
    send_control(steering_angle, throttle)


@sio.on('connect')  # message, disconnect
def connect(sid, environ):
    print("Connected")
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer',
             data={
                 'steering_angle': steering_angle.__str__(),
                 'throttle': throttle.__str__()
             })


if __name__ == '__main__':
    model = keras.models.load_model("nvidia_net_model.h5")

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
