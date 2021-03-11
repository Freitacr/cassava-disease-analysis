from typing import Tuple
import argparse
import sys

import model_training_utils


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
    parser.add_argument("--normalize", required=False, dest="n", action="store_true", default=False)
    return parser


def read_arguments() -> Tuple[str, str, bool, int, int, Tuple[int, int], int, int, str, bool, bool]:
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
    normalize = args.n
    return model_path, out_path, create_new, epochs, batches, \
           input_shape, train_images, val_images, model, summarize, normalize


if __name__ == '__main__':
    model_path, out_path, create_new, epochs, batches, \
        input_shape, train_images, val_images, model, summarize, normalize = read_arguments()
    input_shape = (input_shape[0], input_shape[1], 3)
    if create_new:
        # As opencv2 considers the number of rows to be the second element of a shape tuple,
        #   this is unfortunately necessary.
        model = model_training_utils.create_model(model, (input_shape[1], input_shape[0], input_shape[2]))
    else:
        model = model_training_utils.load_model(model_path)
    if summarize:
        model.summary()
        exit(0)
    model_training_utils.train_model(model, batches, epochs, input_shape[:2],
                                     train_images_per_batch=train_images,
                                     val_images_per_batch=val_images,
                                     normalize_images=normalize)
    model_training_utils.save_model(model, out_path)
