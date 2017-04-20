from PIL import Image
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K
K.set_image_dim_ordering('tf')

import numpy as np
import os

def load_network(weights_path='', img_width=0, img_height=0, color_mode='grayscale'):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, 1 if color_mode == 'grayscale' else 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    model.load_weights(weights_path)

    return model

COLOR_MODE = 'grayscale'
TRY_PATH = os.path.normpath('data/img_try')
WEIGHTS_PATH_FACE = os.path.normpath('data/weights/face.hdf5')
WEIGHTS_PATH_GENDER = os.path.normpath('data/weights/gender.hdf5')
WEIGHTS_PATH_GLASSES = os.path.normpath('data/weights/glasses.hdf5')
WEIGHTS_PATH_GLASSES_T = os.path.normpath('data/weights/type_glasses.hdf5')
IMAGE_PATH = os.path.normpath('data/img_try/1.jpg')
IMG_WIDTH, IMG_HEIGHT = 100, 100

img = Image.open(IMAGE_PATH)
img = img.resize((IMG_WIDTH, IMG_HEIGHT))
img = img.convert('L' if COLOR_MODE == 'grayscale' else 'RGB')
img_try = np.asarray(img, dtype='float32')
img_try = img_try.reshape((IMG_WIDTH, IMG_HEIGHT, 1 if COLOR_MODE == 'grayscale' else 3))
img_try = np.expand_dims(img_try, axis=0)

#init face model
'''
face = Sequential()

face.add(Convolution2D(256, 3, 3, input_shape=(19, 19, 1)))
face.add(Activation('relu'))
face.add(MaxPooling2D())

face.add(Convolution2D(256, 3, 3))
face.add(Activation('relu'))
face.add(MaxPooling2D())

face.add(Flatten())
face.add(Dense(64))
face.add(Activation('relu'))
face.add(Dropout(0.5))
face.add(Dense(1))
face.add(Activation('sigmoid'))

face.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

face.load_weights(WEIGHTS_PATH_FACE)
#

if face.predict(img_try) == [[0.]]:
    print('non-face')
else:
    print('face')
'''

gender = load_network(
    weights_path=WEIGHTS_PATH_GENDER,
    img_width=IMG_WIDTH,
    img_height=IMG_HEIGHT,
    color_mode=COLOR_MODE)

glasses = load_network(
    weights_path=WEIGHTS_PATH_GLASSES,
    img_width=IMG_WIDTH,
    img_height=IMG_HEIGHT,
    color_mode=COLOR_MODE)

type_glasses = load_network(
    weights_path=WEIGHTS_PATH_GLASSES_T,
    img_width=IMG_WIDTH,
    img_height=IMG_HEIGHT,
    color_mode=COLOR_MODE)

if gender.predict(img_try) == [[0.]]:
    print('male')
else:
    print('female')

if glasses.predict(img_try) == [[0.]]:
    if type_glasses.predict(img_try) == [[0.]]:
        print('sunglasses')
    else:
        print('glasses')
else:
    print('clean')