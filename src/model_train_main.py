from typing import Tuple, Dict, Callable, Optional
import argparse
import os
import sys

import keras

import cnn_model
import non_standard_models
import image_utils


__model_string_mapping: Dict[str, Callable[[Tuple[int, int, int], Optional[bool]], keras.Model]] = {
    "cnn": cnn_model.create_model,
    "reimage": non_standard_models.create_reimaging_model,
    "recurrent": non_standard_models.create_recurrent_model
}


def create_model(model_id: str, input_shape: Tuple[int, int, int]) -> keras.Model:
    return __model_string_mapping[model_id](input_shape)


def train_model(model: keras.Model, num_batches: int, epochs_per_batch: int, image_size: Tuple[int, int],
                train_images_per_batch: int = 1000, val_images_per_batch: int = 400) -> None:
    for _ in range(num_batches):
        train_images, train_labels = image_utils.load_batch("train_images", train_images_per_batch, image_size)
        val_images, val_labels = image_utils.load_batch("val_images", val_images_per_batch, image_size)
        model.fit(train_images, train_labels, epochs=epochs_per_batch,
                  validation_data=(val_images, val_labels))


def save_model(model: keras.Model, file_path: str):
    directory, file = os.path.split(file_path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    model.save(file_path)


def load_model(file_path: str) -> keras.Model:
    # let keras raise the error if the file doesn't exist.
    return keras.models.load_model(file_path)


def setup_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", required=False, dest="create", action='store_true', default=False)
    parser.add_argument("model_path", type=str)
    parser.add_argument("-o", required=False, dest="output", type=str)
    parser.add_argument("--epochs", required=False, dest="epochs", type=int, default=100)
    parser.add_argument("--batches", required=False, dest="batches", type=int, default=100)
    parser.add_argument("--rows", required=False, dest="rows", type=int, default=600)
    parser.add_argument("--cols", required=False, dest="cols", type=int, default=400)
    parser.add_argument("--train_images", required=False, dest="train_images", type=int, default=1000)
    parser.add_argument("--val_images", required=False, dest="val_images", type=int, default=400)
    parser.add_argument("--model", required=False, dest="model", type=str, default="cnn")
    parser.add_argument("-s", required=False, dest="summarize", action='store_true', default=False)
    return parser


def read_arguments() -> Tuple[str, str, bool, int, int, Tuple[int, int], int, int, str, bool]:
    args = sys.argv[1:]
    arg_parser = setup_arguments()
    args = arg_parser.parse_args(args)
    model_path = args.model_path
    create_new = args.create
    out_path = model_path
    if hasattr(args, 'output') and args.output is not None:
        out_path = args.output
    epochs = args.epochs
    batches = args.batches
    input_shape = args.rows, args.cols
    train_images = args.train_images
    val_images = args.val_images
    model = args.model
    summarize = args.summarize
    return model_path, out_path, create_new, epochs, batches, input_shape, train_images, val_images, model, summarize


if __name__ == '__main__':
    model_path, out_path, create_new, epochs, batches, \
        input_shape, train_images, val_images, model, summarize = read_arguments()
    input_shape = (input_shape[0], input_shape[1], 3)
    if create_new:
        # As opencv2 considers the number of rows to be the second element of a shape tuple,
        #   this is unfortunately necessary.
        model = create_model(model, (input_shape[1], input_shape[0], input_shape[2]))
    else:
        model = load_model(model_path)
    if summarize:
        model.summary()
        exit(0)
    train_model(model, batches, epochs, input_shape[:2],
                train_images_per_batch=train_images, val_images_per_batch=val_images)
    save_model(model, out_path)
