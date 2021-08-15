from preprocessing import config_preprocessing
import os
import random
import numpy as np
import shutil
from preprocessing.mm2.normalization import NormalizationMM2


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
    split_training_testing_set(config_preprocessing.MM2, percentage=80)
    norm_training = NormalizationMM2(config_preprocessing.MM2['training_set'])
    norm_training.normalize()
    norm_testing = NormalizationMM2(config_preprocessing.MM2['testing_set'])
    norm_testing.normalize()

    return None
