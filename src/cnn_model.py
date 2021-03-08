from typing import Tuple

import keras.layers
import keras.optimizers
import keras.metrics


def create_model(input_shape: Tuple[int, int, int], output_summary: bool = False) -> keras.Sequential:
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(8, 3, input_shape=input_shape, activation='relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(16, 2, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(4, 4)))
    model.add(keras.layers.Dropout(.5))
    model.add(keras.layers.Conv2D(32, 1, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(8, 8)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Dense(5, activation="softmax"))
    model.compile(
        keras.optimizers.rmsprop(),
        loss='categorical_crossentropy',
        metrics=["acc", keras.metrics.categorical_accuracy]
    )
    if output_summary:
        model.summary()
    return model
