import tensorflow as tf
from .data import DataUNet
import sys
from preprocessing import config_preprocessing
sys.path.append('../preprocessing')


class TestingUnet:

    def __init__(self, config):

        self.save_path = config['saved_model']['unet']

    def test_model(self, testing_set):
        trained_model = tf.keras.models.load_model(self.save_path)
        loss, accuracy = trained_model.evaluate(testing_set[0], testing_set[1], verbose=2)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * accuracy))
        print('Restored model, loss: {:5.2f}%'.format(100 * loss))
        # print(trained_model.predict(testing_set[0]).shape)


def test_on_acdc():
    testing_set = DataUNet(
        config_preprocessing.ACDC['id'],
        config_preprocessing.ACDC['testing_set']['path_normalized_images'],
        config_preprocessing.ACDC['testing_set']['path_normalized_masks'],
        config_preprocessing.ACDC['image_size']
    )[0]
    testing_unet = TestingUnet(config_preprocessing.ACDC)
    testing_unet.test_model(testing_set=testing_set)
