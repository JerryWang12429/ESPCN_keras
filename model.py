import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import PIL


def get_model(upscale_factor=3, channels=1):

    inputs = keras.Input(shape=(None, None, channels))

    x = layers.Conv2D(filters=64, kernel_size=5, padding="same", kernel_initializer="Orthogonal", activation="tanh")(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer="Orthogonal", activation="tanh")(x)
    x = layers.Conv2D(filters=channels * (upscale_factor ** 2), kernel_size=3, padding="same", kernel_initializer="Orthogonal", activation="sigmoid")(x)
    outputs = tf.nn.depth_to_space(x, block_size=upscale_factor, data_format='NHWC')

    return keras.Model(inputs, outputs)
