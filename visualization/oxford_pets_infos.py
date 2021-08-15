from preprocessing import config_preprocessing
import os


def nb_images():
    """
    Get the number of images with a corresponding label in the Oxford Pets dataset.
    :return:
    """
    images = os.listdir(config_preprocessing.OXFORD_PETS['images_raw_path'])

    return len(images)
