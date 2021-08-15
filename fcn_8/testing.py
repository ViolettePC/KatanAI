import tensorflow as tf
from .data import DataFCN8
from preprocessing import config_preprocessing
import numpy as np
from scipy.spatial.distance import directed_hausdorff


class TestingFCN8:

    def __init__(self, trained_on):

        if trained_on == 0:
            self.save_path = config_preprocessing.ACDC['saved_model']['fcn_8']
        elif trained_on == 2:
            self.save_path = config_preprocessing.MM2['saved_model']['fcn_8']
        elif trained_on == 1:
            self.save_path = config_preprocessing.OXFORD_PETS['saved_model']['fcn_8']

    def dice(self, ground_truth, predicted_mask):
        """
        Compute DICE coefficient for a given predicted mask with the corresponding ground truth segmentation.
        :param ground_truth: numpy array of the ground truth segmentation.
        :param predicted_mask: numpy array of the predicted mask.
        :return: float, DICE value.
        """
        intersection = 0
        for i in range(ground_truth.shape[0]):
            for j in range(ground_truth.shape[1]):
                if ground_truth[i][j] == predicted_mask[i][j]:
                    intersection += 1
        len_gt = ground_truth.shape[0] * ground_truth.shape[1]
        len_pred = predicted_mask.shape[0] * predicted_mask.shape[1]
        dice = (2 * intersection / (len_gt + len_pred))

        return dice

    def get_dice_values(self, prediction_values, masks):
        """
        Loop on every prediction and call the computation of their DICE values.
        :param prediction_values: numpy arrays of predicted masks.
        :param masks: numpy arrays of ground truth segmentations.
        :return: array containing every DICE values in order.
        """
        dice_values = []
        if len(prediction_values) != len(masks):
            print('Error raised: dice computation')
        for i in range(len(prediction_values)):
            dice_values.append(self.dice(masks[i], prediction_values[i]))

        return dice_values

    def get_hausdorff_values(self, prediction_values, masks):
        """
        Compute Hausdorff Distance for every predicted mask.
        :param prediction_values: numpy array of predicted masks.
        :param masks: numpy array of ground truth segmentations.
        :return: array containing every Hausdorff Distances in order.
        """
        hausdorff_values = []
        if len(prediction_values) != len(masks):
            print('Error raised: hausdorff distance computation')
        for i in range(len(prediction_values)):
            prediction_shape = prediction_values[i].shape
            prediction_value = np.reshape(prediction_values[i], (prediction_shape[0], prediction_shape[1]))
            distance = max(directed_hausdorff(prediction_value, masks[i])[0],
                           directed_hausdorff(masks[i], prediction_value)[0])
            hausdorff_values.append(distance)

        return hausdorff_values

    def test_model(self, testing_set):
        """
        Get loss and accuracy of the trained model when tested on the test set.
        :param testing_set: testing set object
        :return: loss (float), accuracy (float)
        """
        trained_model = tf.keras.models.load_model(self.save_path / 'fcn_8.h5')
        loss, accuracy = trained_model.evaluate(testing_set[0], testing_set[1], verbose=2)

        return loss, accuracy

    def filter_probabilities_predictions(self, prediction_values, min_prob=0.1):
        """
        Filter predicted masks by level of classification confidence.
        :param prediction_values: array of predicted masks.
        :param min_prob: float, minimum confidence.
        :return: array of filtered masks.
        """
        filtred_values = []
        for prection in prediction_values:
            filtred_values.append(np.array(prection > min_prob))

        return filtred_values

    def get_predictions(self, testing_set):
        """
        Get predictions of the trained model on the testing set.
        :param testing_set: testing set object.
        :return: array of predicted masks.
        """
        trained_model = tf.keras.models.load_model(self.save_path / 'fcn_8.h5', compile=True)
        prediction_values = self.filter_probabilities_predictions(
            trained_model.predict(testing_set[0]))

        return prediction_values


def test_on_acdc(trained_on):
    """
    Test trained model on ACDC testing set.
    :param trained_on: int, Id of the dataset used for training.
    :return: loss and accuracy values (floats).
    """
    testing_set = DataFCN8(
        config_preprocessing.ACDC['id'],
        config_preprocessing.ACDC['testing_set']['path_normalized_images'],
        config_preprocessing.ACDC['testing_set']['path_normalized_masks'],
        config_preprocessing.ACDC['image_size']
    )[0]
    testing_fcn8 = TestingFCN8(trained_on)
    loss, accuracy = testing_fcn8.test_model(testing_set=testing_set)

    return loss, accuracy


def test_on_mm2(trained_on):
    """
    Test trained model on M&Ms-2 testing set.
    :param trained_on: int, Id of the dataset used for training.
    :return: loss and accuracy values (floats).
    """
    testing_set = DataFCN8(
        config_preprocessing.MM2['id'],
        config_preprocessing.MM2['testing_set']['path_normalized_images'],
        config_preprocessing.MM2['testing_set']['path_normalized_masks'],
        config_preprocessing.MM2['image_size']
    )[0]
    testing_fcn8 = TestingFCN8(trained_on)
    loss, accuracy = testing_fcn8.test_model(testing_set=testing_set)

    return loss, accuracy


def test_on_oxford_pets(trained_on):
    """
    Test trained model on Oxford Pets testing set.
    :param trained_on: int, Id of the dataset used for training.
    :return: loss and accuracy values (floats).
    """
    testing_set = DataFCN8(
        config_preprocessing.OXFORD_PETS['id'],
        config_preprocessing.OXFORD_PETS['testing_set']['path_normalized_images'],
        config_preprocessing.OXFORD_PETS['testing_set']['path_normalized_masks'],
        config_preprocessing.OXFORD_PETS['image_size']
    )[0]
    testing_fcn8 = TestingFCN8(trained_on)
    loss, accuracy = testing_fcn8.test_model(testing_set=testing_set)

    return loss, accuracy


def predict_on_acdc(trained_on):
    """
    Get predictions of the trained model on the ACDC dataset.
    Compute evaluation metrics (DICE + Hausdorff Distance).
    :param trained_on: int, Id of the dataset used for training.
    :return: array of prediction values, testing set object, array of DICE values, array of HD values.
    """
    testing_set = DataFCN8(
        config_preprocessing.ACDC['id'],
        config_preprocessing.ACDC['testing_set']['path_normalized_images'],
        config_preprocessing.ACDC['testing_set']['path_normalized_masks'],
        config_preprocessing.ACDC['image_size']
    )[0]
    testing_fcn8 = TestingFCN8(trained_on)
    prediction_values = testing_fcn8.get_predictions(testing_set=testing_set)
    dice_values = testing_fcn8.get_dice_values(prediction_values, testing_set[1])
    hausdorff_distance_values = testing_fcn8.get_hausdorff_values(prediction_values, testing_set[1])

    return prediction_values, testing_set, dice_values, hausdorff_distance_values


def predict_on_mm2(trained_on):
    """
    Get predictions of the trained model on the M&Ms-2 dataset.
    Compute evaluation metrics (DICE + Hausdorff Distance).
    :param trained_on: int, Id of the dataset used for training.
    :return: array of prediction values, testing set object, array of DICE values, array of HD values.
    """
    testing_set = DataFCN8(
        config_preprocessing.MM2['id'],
        config_preprocessing.MM2['testing_set']['path_normalized_images'],
        config_preprocessing.MM2['testing_set']['path_normalized_masks'],
        config_preprocessing.MM2['image_size']
    )[0]
    testing_fcn8 = TestingFCN8(trained_on)
    prediction_values = testing_fcn8.get_predictions(testing_set=testing_set)
    dice_values = testing_fcn8.get_dice_values(prediction_values, testing_set[1])
    hausdorff_distance_values = testing_fcn8.get_hausdorff_values(prediction_values, testing_set[1])

    return prediction_values, testing_set, dice_values, hausdorff_distance_values


def predict_on_oxford_pets(trained_on):
    """
    Get predictions of the trained model on the Oxford Pets dataset.
    Compute evaluation metrics (DICE + Hausdorff Distance).
    :param trained_on: int, Id of the dataset used for training.
    :return: array of prediction values, testing set object, array of DICE values, array of HD values.
    """
    testing_set = DataFCN8(
        config_preprocessing.OXFORD_PETS['id'],
        config_preprocessing.OXFORD_PETS['testing_set']['path_normalized_images'],
        config_preprocessing.OXFORD_PETS['testing_set']['path_normalized_masks'],
        config_preprocessing.OXFORD_PETS['image_size']
    )[0]
    testing_fcn8 = TestingFCN8(trained_on)
    prediction_values = testing_fcn8.get_predictions(testing_set=testing_set)
    dice_values = testing_fcn8.get_dice_values(prediction_values, testing_set[1])
    hausdorff_distance_values = testing_fcn8.get_hausdorff_values(prediction_values, testing_set[1])

    return prediction_values, testing_set, dice_values, hausdorff_distance_values
