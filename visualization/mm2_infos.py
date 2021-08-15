import os
from preprocessing import config_preprocessing
from visualization import nifti
import numpy as np


def nb_of_images():
    """
    Get the number of short axis images (both end-systole and end-diastole)
    :return: int, the count of images
    """
    img_count = 0
    patients = os.listdir(config_preprocessing.MM2['raw_path'])
    for patient in patients:
        files = os.listdir(config_preprocessing.MM2['raw_path'] / patient)
        for file in files:
            if file[4:10] == 'SA_ED.' or file[4:10] == 'SA_ES.':
                img_path = config_preprocessing.MM2['raw_path'] / patient / file
                depth = nifti.get_dimensions(img_path)[2]
                img_count += depth

    return img_count


def min_max_slices_per_patients():
    """
    Get the maximum and minimum number of slices per patients.
    :return: min, max (int) values.
    """
    nb_slices = []
    patients = os.listdir(config_preprocessing.MM2['raw_path'])
    for patient in patients:
        files = os.listdir(config_preprocessing.MM2['raw_path'] / patient)
        for file in files:
            if file[4:10] == 'SA_ED.' or file[4:10] == 'SA_ES.':
                img_path = config_preprocessing.MM2['raw_path'] / patient / file
                depth = nifti.get_dimensions(img_path)[2]
                nb_slices.append(depth)

    min, max = np.min(nb_slices), np.max(nb_slices)

    return min, max
