import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import src.relabel_data as rd
import cv2
import numpy as np
from src.create_augement import createAugment
import matplotlib.image as mpimg
from keras.utils import CustomObjectScope
from keras.models import load_model
from PIL import Image, ImageDraw
import random

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# get the new data after filtering and relabeling
x_train, y_train, x_test, y_test = rd.filter_relabel(x_train, y_train, x_test, y_test)

train_masked = createAugment(x_train, x_train)
test_masked = createAugment(x_test, x_test)

## Metric
def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + 1)

def unet_improved():
    inputs = keras.layers.Input((32, 32, 3))

    # Encoding / Downsampling
    conv1 = keras.layers.Conv2D(64, (3, 3), activation=keras.layers.LeakyReLU(), padding='same')(inputs)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Dropout(0.1)(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = keras.layers.Conv2D(128, (3, 3), activation=keras.layers.LeakyReLU(), padding='same')(pool1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Dropout(0.1)(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = keras.layers.Conv2D(256, (3, 3), activation=keras.layers.LeakyReLU(), padding='same')(pool2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Dropout(0.1)(conv3)

    # Decoding / Upsampling
    up4 = keras.layers.concatenate([keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
    conv4 = keras.layers.Conv2D(128, (3, 3), activation=keras.layers.LeakyReLU(), padding='same')(up4)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Dropout(0.1)(conv4)

    up5 = keras.layers.concatenate([keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4), conv1], axis=3)
    conv5 = keras.layers.Conv2D(64, (3, 3), activation=keras.layers.LeakyReLU(), padding='same')(up5)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Dropout(0.1)(conv5)

    # Output
    outputs = keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(conv5)

    return keras.models.Model(inputs=[inputs], outputs=[outputs])


keras.backend.clear_session()
model = unet_improved()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])

model.fit(train_masked, validation_data=test_masked, 
          epochs=200, 
          steps_per_epoch=len(train_masked), 
          validation_steps=len(test_masked),
          use_multiprocessing=True, callbacks=[early_stopping])



model.save('improved_Unet.h5')