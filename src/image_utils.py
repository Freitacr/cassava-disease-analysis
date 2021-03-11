from typing import Tuple, List
import os
from os.path import sep

import cv2
import numpy as np


def __open_jpg(file_path: str, image_size: Tuple[int, int, int]) -> np.ndarray:
    ret = cv2.imread(file_path)
    ret = cv2.resize(ret, image_size[:2])
    return ret


def load_specified_batch(file_paths: List[str], image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
        Loads the specified images, and mapping them to their disease codes.
        :param file_paths: Paths of images to load
        :param image_size: Size the images should be after loading.
        :return: Two numpy arrays containing the loaded image data and disease codes respectively.
            These arrays are index locked; meaning the disease code vector for an image at index i in the image data array
            will be at index i in the disease code array.
        """
    ret_labels = np.zeros((len(file_paths), 5), dtype=np.float32)
    ret_images = np.zeros((len(file_paths), image_size[1], image_size[0], 3))
    for index, file in enumerate(file_paths):
        disease_code_index = int(file.split('_')[-1].split('.')[0])
        ret_labels[index] = np.zeros(5)
        ret_labels[index][disease_code_index] = 1
        ret_images[index] = __open_jpg(file, (*image_size, 3))
        ret_images[index] /= 255
    return ret_images, ret_labels


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
    selected_files = [directory_path + sep + x for x in selected_files]
    return load_specified_batch(selected_files, image_size)
