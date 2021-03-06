import os
import random
import shutil
from .normalization import NormalizationACDC
from preprocessing import config_preprocessing
import numpy as np


def split_training_testing_set(config, percentage):
    """
    Split the raw directory in a training and a testing set.
    :param config: json, access path.
    :param percentage: int, percentage of the raw directory given to the training set.
    :return: None
    """
    patients = os.listdir(config['raw_path'])
    nb_patient = 0
    for _ in patients:
        nb_patient += 1
    testing_set_count = round(nb_patient - ((nb_patient / 100) * percentage))
    testing_set_images = random.sample(os.listdir(config['raw_path']), testing_set_count)
    training_set_images = np.setdiff1d(patients, testing_set_images)

    create_directory(config['testing_set']['path'])
    create_directory(config['training_set']['path'])

    for patient in testing_set_images:
        shutil.copytree(config['raw_path'] / patient, config['testing_set']['path'] / patient)
    for patient in training_set_images:
        shutil.copytree(config['raw_path'] / patient, config['training_set']['path'] / patient)

    return None


def create_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    return None


def launch_preprocessing():
    """
    Preprocessing pipeline, launching normalization for both training and testing sets.
    :return: None
    """
    split_training_testing_set(config_preprocessing.ACDC, percentage=80)
    norm_training = NormalizationACDC(config_preprocessing.ACDC['training_set'])
    norm_training.normalize()
    norm_testing = NormalizationACDC(config_preprocessing.ACDC['testing_set'])
    norm_testing.normalize()

    return None
