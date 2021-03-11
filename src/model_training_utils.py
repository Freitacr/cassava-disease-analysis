from typing import Tuple, Dict, Callable, Optional
import os

import keras
import numpy as np
import tqdm

import cnn_model
import image_utils
import non_standard_models

__model_string_mapping: Dict[str, Callable[[Tuple[int, int, int], Optional[bool]], keras.Model]] = {
    "cnn": cnn_model.create_model,
    "reimage": non_standard_models.create_reimaging_model,
    "recurrent": non_standard_models.create_recurrent_model
}


def create_model(model_id: str, input_shape: Tuple[int, int, int]) -> keras.Model:
    return __model_string_mapping[model_id](input_shape)


def custom_validation(model: keras.Model, image_size: Tuple[int, int], val_images_per_batch: int = 400) -> float:
    val_image_files = os.listdir("val_images")
    curr_index = 0
    num_correct = 0
    num_iterations = (len(val_image_files) // val_images_per_batch) + 1
    print("Custom Validation Batches:")
    for _ in tqdm.tqdm(range(num_iterations)):
        batch_image_paths = val_image_files[curr_index: curr_index + val_images_per_batch]
        batch_image_paths = ["val_images" + os.path.sep + x for x in batch_image_paths]
        batch_images, batch_labels = image_utils.load_specified_batch(batch_image_paths, image_size)
        predicted_labels = model.predict(batch_images)
        for index, predicted in enumerate(predicted_labels):
            predicted_label = np.argmax(predicted)
            actual_label = np.argmax(batch_labels[index])
            if predicted_label == actual_label:
                num_correct += 1
        curr_index += val_images_per_batch
    return num_correct / len(val_image_files)


def train_model(model: keras.Model, num_batches: int, epochs_per_batch: int, image_size: Tuple[int, int],
                train_images_per_batch: int = 1000, val_images_per_batch: int = 400) -> None:
    for _ in range(num_batches):
        train_images, train_labels = image_utils.load_batch("train_images", train_images_per_batch, image_size)
        # val_images, val_labels = image_utils.load_batch("val_images", val_images_per_batch, image_size)
        for _ in range(epochs_per_batch):
            model.fit(train_images, train_labels, epochs=1)
            print(f"Calculated val_acc: {custom_validation(model, image_size, val_images_per_batch)}")


def save_model(model: keras.Model, file_path: str):
    directory, file = os.path.split(file_path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    model.save(file_path)


def load_model(file_path: str) -> keras.Model:
    # let keras raise the error if the file doesn't exist.
    return keras.models.load_model(file_path)