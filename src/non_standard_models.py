from typing import Tuple

import keras.layers
import keras.optimizers
import keras.metrics


def create_reimaging_model(input_shape: Tuple[int, int, int], output_summary: bool = False) -> keras.Sequential:
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(8, 3, input_shape=input_shape, activation='relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(16, 2, activation='elu'))
    model.add(keras.layers.MaxPool2D(pool_size=(4, 4)))
    model.add(keras.layers.GaussianNoise(.06))
    model.add(keras.layers.Conv2D(32, 1, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(8, 8)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(12288))
    model.add(keras.layers.Reshape((64, 64, 3)))
    model.add(keras.layers.Conv2D(16, 2, activation='elu'))
    model.add(keras.layers.MaxPool2D(pool_size=(4, 4)))
    model.add(keras.layers.Dropout(.5))
    model.add(keras.layers.Conv2D(32, 1, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(8, 8)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(5, activation='softmax'))
    model.compile(
        keras.optimizers.rmsprop(),
        loss='categorical_crossentropy',
        metrics=["acc", keras.metrics.categorical_accuracy]
    )
    if output_summary:
        model.summary()
    return model


def create_recurrent_model(input_shape: Tuple[int, int, int], output_summary: bool = False) -> keras.Sequential:
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(16, 8, input_shape=input_shape, activation='relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Reshape((16, 8)))
    model.add(keras.layers.LSTM(48, return_sequences=True))
    model.add(keras.layers.Reshape((16, 16, 3)))
    model.add(keras.layers.Conv2D(8, 1, activation='elu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(16, 2, activation='elu'))
    model.add(keras.layers.MaxPool2D(pool_size=(4, 4)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(5, activation='softmax'))
    model.compile(
        keras.optimizers.rmsprop(),
        loss='categorical_crossentropy',
        metrics=["acc", keras.metrics.categorical_accuracy]
    )
    if output_summary:
        model.summary()
    return model
