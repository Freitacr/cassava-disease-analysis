from typing import Dict, Tuple, List
import os
from os.path import sep
import random


def __move_images(image_file_names: List[str],
                  initial_directory: str,
                  destination_directory: str,
                  image_disease_mapping: Dict[str, str]) -> None:
    initial_files = [initial_directory + sep + x for x in image_file_names]

    for i in range(len(image_file_names)):
        initial_file_name = image_file_names[i]
        initial_file_path = initial_files[i]
        destination_file_name = initial_file_name
        if '_' not in initial_file_name:
            disease_code = image_disease_mapping[initial_file_name]
            destination_file_name = initial_file_name.replace('.jpg', f'_{disease_code}.jpg')
        destination_file_path = destination_directory + sep + destination_file_name
        with open(initial_file_path, 'rb') as initial_handle:
            with open(destination_file_path, 'wb') as destination_handle:
                destination_handle.write(initial_handle.read())


def split_images(train_directory: str,
                 test_directory: str,
                 val_directory: str,
                 image_disease_mapping: Dict[str, str],
                 split_percentages: Tuple[float, float, float]) -> None:
    """
    Splits the .jpg images within train_directory into train_-, val_-, and test_directory based on
        the percentages provided
    :param train_directory: Initial directory containing all images that need to be split.
        After splitting, this directory will contain all images that are for the training data set.
    :param test_directory: After splitting, this directory will contain all images that are for the test data set.
    :param val_directory: After splitting, this directory will contain all images that are for the validation data set.
    :param image_disease_mapping: Dictionary mapping file names to the disease code associated with the file.
    :param split_percentages: Tuple containing the percentages of images that should end up within each of the
        directories.

    :example:
        split_images('train_images', 'test_images', 'val_images', disease_mapping, (.6, .25, .15))
        The above call would move 25% of the original images from 'train_images' into 'test_images', and 15% would
            be moved into 'val_images', appending _{disease_code} prior to the extension for all of the images.
            disease_code is the mapped code in the image_disease_mapping for the current file.
    :return: None
    """
    file_names = os.listdir(train_directory)
    random.shuffle(file_names)
    num_test_files = int(len(file_names) * split_percentages[1])
    num_val_files = int(len(file_names) * split_percentages[2])
    test_files = file_names[:num_test_files]
    val_files = file_names[num_test_files:num_test_files+num_val_files]
    train_files = file_names[num_val_files+num_test_files:]
    __move_images(test_files, train_directory, test_directory, image_disease_mapping)
    __move_images(val_files, train_directory, val_directory, image_disease_mapping)
    __move_images(train_files, train_directory, train_directory, image_disease_mapping)
    for path in [train_directory + sep + x for x in file_names]:
        os.remove(path)


def load_image_disease_mapping(file_path: str) -> Dict[str, str]:
    """
    Loads the stored mapping of file names to disease codes from the csv file specified.
    :param file_path: Path to the csv file containing the mapping from file names to disease codes
    :return: Dictionary containing the mapping of image file names to disease codes.
    """
    ret = {}
    with open(file_path, 'r') as handle:
        handle.readline()  # ignore heading row
        for line in handle:
            line_split = line.split(',')
            ret[line_split[0].strip()] = line_split[1].strip()
    return ret


if __name__ == '__main__':
    image_disease_mapping = load_image_disease_mapping("train.csv")
    split_images("train_images", "test_images", "val_images", image_disease_mapping, (.60, .25, .15))
