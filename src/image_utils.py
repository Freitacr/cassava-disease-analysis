from typing import Tuple, List, Dict
import os
from os.path import sep

import cv2
import numpy as np


__image_files_by_label: Dict[int, List[str]] = {}
__num_label_categories: int = 5


def __open_jpg(file_path: str, image_size: Tuple[int, int, int]) -> np.ndarray:
    ret = cv2.imread(file_path)
    ret = cv2.resize(ret, image_size[:2])
    return ret


def __load_image_file_mapping(directory_path: str):
    if __image_files_by_label:
        return
    image_files = os.listdir(directory_path)
    for image_file in image_files:
        disease_code_index = int(image_file.split('_')[-1].split('.')[0])
        if disease_code_index not in __image_files_by_label:
            __image_files_by_label[disease_code_index] = []
        __image_files_by_label[disease_code_index].append(directory_path + sep + image_file)


def load_specified_batch(file_paths: List[str],
                         image_size: Tuple[int, int], normalize: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
        Loads the specified images, and mapping them to their disease codes.
        :param file_paths: Paths of images to load
        :param image_size: Size the images should be after loading.
        :param normalize: Flag for image normalization. If true, all elements in an image are divided by 255
            to translate them into the range [0.0-1.0]
        :return: Two numpy arrays containing the loaded image data and disease codes respectively.
            These arrays are index locked; meaning the disease code vector for an image at index i in the image data array
            will be at index i in the disease code array.
        """
    ret_labels = np.zeros((len(file_paths), __num_label_categories), dtype=np.float32)
    ret_images = np.zeros((len(file_paths), image_size[1], image_size[0], 3))
    for index, file in enumerate(file_paths):
        disease_code_index = int(file.split('_')[-1].split('.')[0])
        ret_labels[index] = np.zeros(__num_label_categories)
        ret_labels[index][disease_code_index] = 1
        ret_images[index] = __open_jpg(file, (*image_size, 3))
        if normalize:
            ret_images[index] /= 255
    return ret_images, ret_labels


def load_batch(directory_path: str, num_to_load: int,
               image_size: Tuple[int, int], normalize: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the specified amount of images from the directory, and mapping them to their disease codes.
    :param directory_path: Path to the directory where images should be loaded from
    :param num_to_load: The number of images to load from the directory.
    :param image_size: Size the images should be after loading.
    :param normalize: Flag for image normalization. If true, all elements in an image are divided by 255
            to translate them into the range [0.0-1.0]
    :return: Two numpy arrays containing the loaded image data and disease codes respectively.
        These arrays are index locked; meaning the disease code vector for an image at index i in the image data array
        will be at index i in the disease code array.
    """
    if not __image_files_by_label:
        __load_image_file_mapping(directory_path)
    num_remaining = num_to_load
    num_per_category = int(num_to_load / __num_label_categories)
    selected_files = []

    for i in range(__num_label_categories-1):
        selected_files.extend(np.random.choice(__image_files_by_label[i], num_per_category))
        num_remaining -= num_per_category
    selected_files.extend(np.random.choice(__image_files_by_label[__num_label_categories-1], num_remaining))
    return load_specified_batch(selected_files, image_size, normalize)
