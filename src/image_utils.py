from typing import Tuple
import os
from os.path import sep

import cv2
import numpy as np


def __open_jpg(file_path: str, image_size: Tuple[int, int, int]) -> np.ndarray:
    ret = cv2.imread(file_path)
    ret = ret.resize(image_size)
    return ret


def load_batch(directory_path: str, num_to_load: int, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the specified amount of images from the directory, and mapping them to their disease codes.
    :param directory_path: Path to the directory where images should be loaded from
    :param num_to_load: The number of images to load from the directory.
    :param image_size: Size the images should be after loading.
    :return: Two numpy arrays containing the loaded image data and disease codes respectively.
        These arrays are index locked; meaning the disease code vector for an image at index i in the image data array
        will be at index i in the disease code array.
    """
    files = os.listdir(directory_path)
    selected_files = np.random.choice(files, num_to_load)
    ret_labels = np.zeros((num_to_load, 5), dtype=np.int32)
    ret_images = np.zeros((num_to_load, *image_size, 3))
    for index, file in enumerate(selected_files):
        file_path = directory_path + sep + file
        disease_code_index = int(file.split('_')[1].split('.')[0])
        ret_labels[index] = np.zeros(5)
        ret_labels[index][disease_code_index] = 1
        ret_images[index] = __open_jpg(file_path, (*image_size, 3))
    return ret_images, ret_labels
