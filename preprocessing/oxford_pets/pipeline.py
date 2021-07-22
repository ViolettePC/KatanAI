import os
import numpy as np
import shutil
import sys
import random
from .normalization import NormalizationOxfordPets
from preprocessing import config_preprocessing
sys.path.append('../preprocessing')


def split_training_testing_set(config, percentage):

    pets_images = os.listdir(config['images_raw_path'])
    pets_masks = os.listdir(config['masks_raw_path'])
    total = check_shape_dataset(pets_images, pets_masks)

    if total is None:
        print('Error raised in oxford_pets dataset while preprocessing')

        return None

    create_directory(config['training_set']['path_images'])
    create_directory(config['training_set']['path_masks'])
    create_directory(config['testing_set']['path_images'])
    create_directory(config['testing_set']['path_masks'])

    testing_set_count = round(total - ((total / 100) * percentage))
    testing_set_images = random.sample(pets_images, testing_set_count)
    training_set_images = np.setdiff1d(pets_images, testing_set_images)

    for image in testing_set_images:
        shutil.copy(config['images_raw_path'] / image, config['testing_set']['path_images'])
        shutil.copy(str(config['masks_raw_path']) + '/' + image[:-4] + '.png', config['testing_set']['path_masks'])

    for image in training_set_images:
        shutil.copy(config['images_raw_path'] / image, config['training_set']['path_images'])
        shutil.copy(str(config['masks_raw_path']) + '/' + image[:-4] + '.png', config['training_set']['path_masks'])

    return None


def check_shape_dataset(pets_images, pets_masks):
    pets_images_count = 0
    pets_masks_count = 0

    for image in pets_images:
        if image[-4:] == '.jpg':
            pets_images_count += 1
    for mask in pets_masks:
        if mask[-4:] == '.png':
            pets_masks_count += 1

    if pets_masks_count != pets_images_count:

        return None

    return pets_images_count


def create_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def launch_preprocessing():
    split_training_testing_set(config_preprocessing.OXFORD_PETS, percentage=80)
    norm_training = NormalizationOxfordPets(config_preprocessing.OXFORD_PETS['training_set'])
    norm_training.normalize()
    norm_testing = NormalizationOxfordPets(config_preprocessing.OXFORD_PETS['testing_set'])
    norm_testing.normalize()

    return None
