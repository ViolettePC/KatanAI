from tensorflow import keras
from .model import unet_model
from .data import DataUNet
import sys
import os
import shutil
from preprocessing import config_preprocessing
sys.path.append('../preprocessing')


class TrainingUnet:

    def __init__(self, config, n_levels):

        self.image_size = config['image_size']
        self.n_levels = n_levels
        self.path_normalized = config['training_set']['path_normalized']
        self.path_normalized_images = config['training_set']['path_normalized_images']
        self.path_normalized_masks = config['training_set']['path_normalized_masks']
        self.save_path = config['saved_model']['unet']

    def build_model(self):
        keras.backend.clear_session()
        model = unet_model(self.image_size, self.n_levels)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

    def train_model(self, num_epochs, n_levels, training_set):
        keras.backend.clear_session()
        model = unet_model(self.image_size, n_levels)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x=training_set[0], y=training_set[1], epochs=num_epochs, shuffle=True, validation_split=0.2)
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.mkdir(self.save_path)
        model.save(self.save_path)


def train_on_acdc():
    training_set = DataUNet(
        config_preprocessing.ACDC['id'],
        config_preprocessing.ACDC['training_set']['path_normalized_images'],
        config_preprocessing.ACDC['training_set']['path_normalized_masks'],
        config_preprocessing.ACDC['image_size']
    )[0]
    training_unet = TrainingUnet(config_preprocessing.ACDC, 4)
    training_unet.train_model(num_epochs=1, n_levels=4, training_set=training_set)


