import os
import random
import shutil
from .normalization import NormalizationACDC
import sys
from preprocessing import config_preprocessing
import numpy as np
sys.path.append('../preprocessing')


def split_training_testing_set(config, percentage):
    patients = os.listdir(config['raw_path'])
    nb_patient = 0
    for patient in patients:
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
    Warning: Need to be launched only once!
    :return:
    """
    split_training_testing_set(config_preprocessing.ACDC, percentage=80)
    norm_training = NormalizationACDC(config_preprocessing.ACDC['training_set'])
    norm_training.normalize()
    norm_testing = NormalizationACDC(config_preprocessing.ACDC['testing_set'])
    norm_testing.normalize()

    return None
