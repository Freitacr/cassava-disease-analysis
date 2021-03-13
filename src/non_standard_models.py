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


def create_granular_model(input_shape: Tuple[int, int, int], output_summary: bool = False) -> keras.Model:
    input_layer = keras.Input(input_shape)

    conv_layer_args = [
        [(4, 6), (8, 6)],
        [(8, 6), (16, 6)],
        [(16, 6), (32, 6)]
    ]

    granular_layer_chains = []

    for granular_args in conv_layer_args:
        layer = keras.layers.Conv2D(*granular_args[0], activation='relu')(input_layer)
        layer = keras.layers.MaxPool2D()(layer)
        layer = keras.layers.Dropout(.5)(layer)
        layer = keras.layers.Conv2D(*granular_args[1], activation='tanh')(layer)
        layer = keras.layers.MaxPool2D()(layer)
        layer = keras.layers.Dropout(.5)(layer)
        layer = keras.layers.Flatten()(layer)
        layer = keras.layers.Dense(32)(layer)
        layer = keras.layers.Dense(32)(layer)
        granular_layer_chains.append(layer)

    layer = keras.layers.Concatenate()([granular_layer_chains[0], granular_layer_chains[1]])
    layer = keras.layers.Concatenate()([layer, granular_layer_chains[2]])
    layer = keras.layers.Dense(32)(layer)
    layer = keras.layers.Dense(128)(layer)
    layer = keras.layers.Dense(64)(layer)
    layer = keras.layers.Dense(5, activation='softmax')(layer)
    model = keras.Model(inputs=input_layer, outputs=layer)
    model.compile(
        keras.optimizers.rmsprop(),
        loss='categorical_crossentropy',
        metrics=["acc", keras.metrics.categorical_accuracy]
    )
    if output_summary:
        model.summary()
    return model
