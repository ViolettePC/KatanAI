from tensorflow import keras
from .model import fcn8_model
from .data import DataFCN8
import os
import shutil
import json
from preprocessing import config_preprocessing


class TrainingFCN8:

    def __init__(self, save_path, image_size):

        self.save_path = save_path
        self.image_size = image_size

    def build_model(self):
        """
        Get model summary.
        :return: None.
        """
        keras.backend.clear_session()
        model = fcn8_model(self.image_size)
        model.summary()

    def train_model(self, num_epochs, training_set):
        """
        Train the model on a specific set.
        Save the trained model's weights.
        Save the training metrics.
        :param num_epochs: int, number of epochs.
        :param training_set: training set object.
        :return: None.
        """
        keras.backend.clear_session()
        model = fcn8_model(self.image_size)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(x=training_set[0],
                            y=training_set[1],
                            epochs=num_epochs,
                            shuffle=True,
                            validation_split=0.2,
                            batch_size=32)

        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.mkdir(self.save_path)
        model.save(self.save_path / 'fcn_8.h5')

        if os.path.isfile(self.save_path / 'history.json'):
            os.remove(self.save_path / 'history.json')
        with open(self.save_path / 'history.json', 'w') as f:
            json.dump(history.history, f)


def train_on_acdc():
    """
    Train the model on ACDC training set.
    :return: None
    """
    training_set = DataFCN8(
        config_preprocessing.ACDC['id'],
        config_preprocessing.ACDC['training_set']['path_normalized_images'],
        config_preprocessing.ACDC['training_set']['path_normalized_masks'],
        config_preprocessing.ACDC['image_size']
    )[0]
    training_fcn_8 = TrainingFCN8(config_preprocessing.ACDC['saved_model']['unet'],
                                  config_preprocessing.ACDC['image_size'])
    training_fcn_8.build_model()
    training_fcn_8.train_model(num_epochs=50, training_set=training_set)


def train_on_mm2():
    """
    Train the model on M&Ms-2 training set.
    :return: None
    """
    training_set = DataFCN8(
        config_preprocessing.MM2['id'],
        config_preprocessing.MM2['training_set']['path_normalized_images'],
        config_preprocessing.MM2['training_set']['path_normalized_masks'],
        config_preprocessing.MM2['image_size']
    )[0]
    training_fcn_8 = TrainingFCN8(config_preprocessing.MM2['saved_model']['unet'],
                                  config_preprocessing.MM2['image_size'])
    training_fcn_8.train_model(num_epochs=50, training_set=training_set)


def train_on_oxford_pets():
    """
    Train the model on Oxford Pets training set.
    :return: None
    """
    training_set = DataFCN8(
        config_preprocessing.OXFORD_PETS['id'],
        config_preprocessing.OXFORD_PETS['training_set']['path_normalized_images'],
        config_preprocessing.OXFORD_PETS['training_set']['path_normalized_masks'],
        config_preprocessing.OXFORD_PETS['image_size']
    )[0]
    training_fcn_8 = TrainingFCN8(config_preprocessing.OXFORD_PETS['saved_model']['unet'],
                                  config_preprocessing.OXFORD_PETS['image_size'])
    training_fcn_8.train_model(num_epochs=50, training_set=training_set)
