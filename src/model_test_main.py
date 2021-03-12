from typing import Tuple, Dict, List
import argparse
import sys

import model_utils


def setup_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--rows", required=False, dest="rows", type=int, default=600)
    parser.add_argument("--cols", required=False, dest="cols", type=int, default=400)
    parser.add_argument("--batch_size", required=False, dest="batch", type=int, default=1000)
    parser.add_argument("--normalize", required=False, dest="n", action="store_true", default=True)
    parser.add_argument("-s", required=False, dest="summarize", action='store_true', default=False)
    return parser


def read_arguments() -> Tuple[str, Tuple[int, int], int, bool, bool]:
    args = sys.argv[1:]
    arg_parser = setup_arguments()
    args = arg_parser.parse_args(args)
    model_path = args.model_path
    rows = args.rows
    cols = args.cols
    batch_size = args.batch
    normalize = args.n
    summarize = args.summarize
    return model_path, (rows, cols), batch_size, normalize, summarize


if __name__ == '__main__':
    model_path, image_size, batch_size, normalize, summarize = read_arguments()
    model = model_utils.load_model(model_path)
    if summarize:
        model.summary()
        exit(0)
    prediction_stats: Dict[int, List[int]] = {}
    accuracy = model_utils.custom_accuracy(model, image_size, normalize,
                                           images_per_batch=batch_size, image_dir="test_images",
                                           prediction_stats_storage=prediction_stats)

    print(f"Model accuracy on Testing Dataset: {accuracy}")
    model_utils.print_prediction_stats(prediction_stats)
