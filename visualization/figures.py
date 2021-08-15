from preprocessing import config_preprocessing
from visualization import nifti
from matplotlib import pyplot as plt
from preprocessing.acdc.normalization import NormalizationACDC
from visualization import predictions
from visualization import model_training_metrics
from u_net import testing as testing_unet
from fcn_8 import testing as testing_fcn8
import numpy as np


def figure_1():
    """
    Presentation of a raw ACDC image and the corresponding raw ground truth segmentation.
    (Patient 1, frame 1, slice 3).
    :return: None
    """
    path_img = config_preprocessing.ACDC['raw_path'] / 'patient001' / 'patient001_frame01.nii.gz'
    path_mask = config_preprocessing.ACDC['raw_path'] / 'patient001' / 'patient001_frame01_gt.nii.gz'
    img_data = nifti.get_slice_in_3d_img(path_img, 3)
    mask_data = nifti.get_slice_in_3d_img(path_mask, 3)

    fig = plt.figure()
    axes = []
    cols, rows = 2, 1

    axes.append(fig.add_subplot(rows, cols, 1))
    plt.imshow(img_data, interpolation='nearest')
    plt.tick_params(labelsize=7)
    plt.xlabel('px', fontsize=7)
    plt.ylabel('px', fontsize=7)

    axes.append(fig.add_subplot(rows, cols, 2))
    plt.imshow(mask_data, interpolation='nearest')
    plt.tick_params(labelsize=7)
    plt.xlabel('px', fontsize=7)
    plt.ylabel('px', fontsize=7)

    fig.tight_layout()
    plt.show()

    return None


def figure_2():
    """
    Presentation of a raw M&Ms-2 image and the corresponding raw ground truth segmentation.
    (Patient 1, frame 1, slice 5).
    :return: None
    """
    path_img = config_preprocessing.MM2['raw_path'] / '001' / '001_SA_ED.nii.gz'
    path_mask = config_preprocessing.MM2['raw_path'] / '001' / '001_SA_ED_gt.nii.gz'
    img_data = nifti.get_slice_in_3d_img(path_img, 5)
    mask_data = nifti.get_slice_in_3d_img(path_mask, 5)

    fig = plt.figure()
    axes = []
    cols, rows = 2, 1

    axes.append(fig.add_subplot(rows, cols, 1))
    plt.imshow(img_data, interpolation='nearest')
    plt.tick_params(labelsize=7)
    plt.xlabel('px', fontsize=7)
    plt.ylabel('px', fontsize=7)

    axes.append(fig.add_subplot(rows, cols, 2))
    plt.imshow(mask_data, interpolation='nearest')
    plt.tick_params(labelsize=7)
    plt.xlabel('px', fontsize=7)
    plt.ylabel('px', fontsize=7)

    fig.tight_layout()
    plt.show()

    return None


def figure_3():
    """
    Illustrate the changes of an ACDC image going though the normalization process.
    An individual figure is create for every step (Raw, Rescaled, Cropped, CLAHE) and the labeled image.
    Patient 1, frame 1, slice 3.
    :return: None
    """
    norm_testing = NormalizationACDC(config_preprocessing.ACDC['testing_set'])
    img_path = config_preprocessing.ACDC['raw_path'] / 'patient001' / 'patient001_frame01.nii.gz'
    images = norm_testing.transform_img_to_numpy_arrays(img_path, 0)
    pixdim = norm_testing.get_spatial_resolution(img_path)
    images_rescaled = norm_testing.rescale(images, pixdim)
    images_cropped = norm_testing.crop(images_rescaled)
    images_clahe = norm_testing.clahe(images_cropped)

    plt.imshow(images[3])
    plt.xlabel('px', fontsize=7)
    plt.ylabel('px', fontsize=7)
    plt.title('Original', fontsize='xx-large', fontweight='heavy')
    plt.show()

    plt.imshow(images_rescaled[3])
    plt.xlabel('px', fontsize=7)
    plt.ylabel('px', fontsize=7)
    plt.title('Rescaled', fontsize='xx-large', fontweight='heavy')
    plt.show()

    plt.imshow(images_cropped[3])
    plt.xlabel('px', fontsize=7)
    plt.ylabel('px', fontsize=7)
    plt.title('Cropped', fontsize='xx-large', fontweight='heavy')
    plt.show()

    plt.imshow(images_clahe[3])
    plt.xlabel('px', fontsize=7)
    plt.ylabel('px', fontsize=7)
    plt.title('CLAHE', fontsize='xx-large', fontweight='heavy')
    plt.show()

    mask_path = config_preprocessing.ACDC['raw_path'] / 'patient001' / 'patient001_frame01_gt.nii.gz'
    masks = norm_testing.transform_img_to_numpy_arrays(mask_path, 1)
    pixdim = norm_testing.get_spatial_resolution(mask_path)
    masks_rescaled = norm_testing.rescale(masks, pixdim)
    masks_cropped = norm_testing.crop(masks_rescaled)

    plt.imshow(masks_cropped[3])
    plt.xlabel('px', fontsize=7)
    plt.ylabel('px', fontsize=7)
    plt.title('Corresponding Label', fontsize='xx-large', fontweight='heavy')
    plt.show()

    return None


def figure_6():
    """
    Plot learning history (accuracy and loss) of FCN-8 on M&Ms-2.
    :return: None
    """
    model_training_metrics.plot_training_accuracy(
        config_preprocessing.MM2['saved_model']['fcn_8'] / 'history_fcn_8.json')
    model_training_metrics.plot_training_loss(
        config_preprocessing.MM2['saved_model']['fcn_8'] / 'history_fcn_8.json')

    return None


def figure_7():
    """
    Figure comparing the ground truth segmentation with the predicted mask and the original image.
    The DICE coefficient and Hausdorff distance values are also added to the figure.
    :return: None
    """
    prediction_values, testing_set, dice_values, hausdorff_distance_values = testing_unet.predict_on_acdc(0)
    predictions.show_prediction(testing_set, prediction_values, dice_values, hausdorff_distance_values, 1)

    return None


def figure_8():
    figure_8_a()
    figure_8_b()
    return None


def figure_8_a():
    """
    Create box whisker plots of DICE values for the 4 experiments using Unet.
    :return: None.
    """
    experiment_unet_1 = testing_unet.predict_on_acdc(0)
    experiment_unet_2 = testing_unet.predict_on_mm2(0)
    experiment_unet_3 = testing_unet.predict_on_mm2(2)
    experiment_unet_4 = testing_unet.predict_on_acdc(2)

    plt.boxplot((experiment_unet_1[2],
                 experiment_unet_2[2],
                 experiment_unet_3[2],
                 experiment_unet_4[2]))
    plt.ylabel('DICE coefficient', fontsize=11)
    plt.xlabel('Experiment number', fontsize=11)
    plt.show()

    return None


def figure_8_b():
    """
    Create box whisker plots of DICE values for the 4 experiments using FCN-8.
    :return: None.
    """
    experiment_fcn8_1 = testing_fcn8.predict_on_acdc(0)
    experiment_fcn8_2 = testing_fcn8.predict_on_mm2(0)
    experiment_fcn8_3 = testing_fcn8.predict_on_mm2(2)
    experiment_fcn8_4 = testing_fcn8.predict_on_acdc(2)

    plt.boxplot((experiment_fcn8_1[2],
                 experiment_fcn8_2[2],
                 experiment_fcn8_3[2],
                 experiment_fcn8_4[2]))
    plt.ylabel('DICE coefficient', fontsize=11)
    plt.xlabel('Experiment number', fontsize=11)
    plt.show()

    return None


def figure_9():
    figure_9_a()
    figure_9_b()
    return None


def figure_9_a():
    """
    Create box whisker plots of Hausdorff distance values for the 4 experiments using Unet.
    :return: None.
    """
    experiment_unet_1 = testing_unet.predict_on_acdc(0)
    experiment_unet_2 = testing_unet.predict_on_mm2(0)
    experiment_unet_3 = testing_unet.predict_on_mm2(2)
    experiment_unet_4 = testing_unet.predict_on_acdc(2)

    plt.boxplot((experiment_unet_1[3],
                 experiment_unet_2[3],
                 experiment_unet_3[3],
                 experiment_unet_4[3]))
    plt.ylabel('Hausdorff distance', fontsize=11)
    plt.xlabel('Experiment number', fontsize=11)
    plt.show()

    return None


def figure_9_b():
    """
    Create box whisker plots of Hausdorff distance values for the 4 experiments using FCN-8.
    :return: None.
    """
    experiment_fcn8_1 = testing_fcn8.predict_on_acdc(0)
    experiment_fcn8_2 = testing_fcn8.predict_on_mm2(0)
    experiment_fcn8_3 = testing_fcn8.predict_on_mm2(2)
    experiment_fcn8_4 = testing_fcn8.predict_on_acdc(2)

    plt.boxplot((experiment_fcn8_1[3],
                 experiment_fcn8_2[3],
                 experiment_fcn8_3[3],
                 experiment_fcn8_4[3]))
    plt.ylabel('Hausdorff distance', fontsize=11)
    plt.xlabel('Experiment number', fontsize=11)
    plt.show()

    return None


def table_3():
    """
    Get and print information for Table 3.
    :return: None
    """
    experiment_unet_1 = testing_unet.predict_on_acdc(0)
    unet_1_loss, unet_1_accuracy = testing_unet.test_on_acdc(0)

    experiment_unet_2 = testing_unet.predict_on_mm2(0)
    unet_2_loss, unet_2_accuracy = testing_unet.test_on_mm2(0)

    experiment_unet_3 = testing_unet.predict_on_mm2(2)
    unet_3_loss, unet_3_accuracy = testing_unet.test_on_mm2(2)

    experiment_unet_4 = testing_unet.predict_on_acdc(2)
    unet_4_loss, unet_4_accuracy = testing_unet.test_on_acdc(2)

    print('unet 1.............................................')
    print('mean dice', round(np.mean(experiment_unet_1[2]), 3))
    print('mean HD', round(np.mean(experiment_unet_1[3]), 3))
    print('accuracy: {:5.2f}%'.format(100 * unet_1_accuracy))
    print('loss: {:5.2f}%'.format(100 * unet_1_loss))
    print('...................................................')

    print('unet 2.............................................')
    print('mean dice', round(np.mean(experiment_unet_2[2]), 3))
    print('mean HD', round(np.mean(experiment_unet_2[3]), 3))
    print('accuracy: {:5.2f}%'.format(100 * unet_2_accuracy))
    print('loss: {:5.2f}%'.format(100 * unet_2_loss))
    print('...................................................')

    print('unet 3.............................................')
    print('mean dice', round(np.mean(experiment_unet_3[2]), 3))
    print('mean HD', round(np.mean(experiment_unet_3[3]), 3))
    print('accuracy: {:5.2f}%'.format(100 * unet_3_accuracy))
    print('loss: {:5.2f}%'.format(100 * unet_3_loss))
    print('...................................................')

    print('unet 4.............................................')
    print('mean dice', round(np.mean(experiment_unet_4[2]), 3))
    print('mean HD', round(np.mean(experiment_unet_4[3]), 3))
    print('accuracy: {:5.2f}%'.format(100 * unet_4_accuracy))
    print('loss: {:5.2f}%'.format(100 * unet_4_loss))
    print('...................................................')

    return None


def table_4():
    """
    Get and print information for Table 4.
    :return:
    """
    experiment_fcn8_1 = testing_fcn8.predict_on_acdc(0)
    fcn8_1_loss, fcn8_1_accuracy = testing_fcn8.test_on_acdc(0)

    experiment_fcn8_2 = testing_fcn8.predict_on_mm2(0)
    fcn8_2_loss, fcn8_2_accuracy = testing_fcn8.test_on_mm2(0)

    experiment_fcn8_3 = testing_fcn8.predict_on_mm2(2)
    fcn8_3_loss, fcn8_3_accuracy = testing_fcn8.test_on_mm2(2)

    experiment_fcn8_4 = testing_fcn8.predict_on_acdc(2)
    fcn8_4_loss, fcn8_4_accuracy = testing_fcn8.test_on_acdc(2)

    print('fcn8 1.............................................')
    print('mean dice', round(np.mean(experiment_fcn8_1[2]), 3))
    print('mean HD', round(np.mean(experiment_fcn8_1[3]), 3))
    print('accuracy: {:5.2f}%'.format(100 * fcn8_1_accuracy))
    print('loss: {:5.2f}%'.format(100 * fcn8_1_loss))
    print('...................................................')

    print('fcn8 2.............................................')
    print('mean dice', round(np.mean(experiment_fcn8_2[2]), 3))
    print('mean HD', round(np.mean(experiment_fcn8_2[3]), 3))
    print('accuracy: {:5.2f}%'.format(100 * fcn8_2_accuracy))
    print('loss: {:5.2f}%'.format(100 * fcn8_2_loss))
    print('...................................................')

    print('fcn8 3.............................................')
    print('mean dice', round(np.mean(experiment_fcn8_3[2]), 3))
    print('mean HD', round(np.mean(experiment_fcn8_3[3]), 3))
    print('accuracy: {:5.2f}%'.format(100 * fcn8_3_accuracy))
    print('loss: {:5.2f}%'.format(100 * fcn8_3_loss))
    print('...................................................')

    print('fcn8 4.............................................')
    print('mean dice', round(np.mean(experiment_fcn8_4[2]), 3))
    print('mean HD', round(np.mean(experiment_fcn8_4[3]), 3))
    print('accuracy: {:5.2f}%'.format(100 * fcn8_4_accuracy))
    print('loss: {:5.2f}%'.format(100 * fcn8_4_loss))
    print('...................................................')

    return None


def figure_b1():
    """
    Learning curves of Appendix B.
    :return: None.
    """
    model_training_metrics.plot_training_accuracy(
        config_preprocessing.ACDC['saved_model']['fcn_8'] / 'history_fcn_8.json')
    model_training_metrics.plot_training_loss(
        config_preprocessing.ACDC['saved_model']['fcn_8'] / 'history_fcn_8.json')

    model_training_metrics.plot_training_accuracy(
        config_preprocessing.MM2['saved_model']['fcn_8'] / 'history_fcn_8.json')
    model_training_metrics.plot_training_loss(
        config_preprocessing.MM2['saved_model']['fcn_8'] / 'history_fcn_8.json')

    model_training_metrics.plot_training_accuracy(
        config_preprocessing.ACDC['saved_model']['unet'] / 'history_unet.json')
    model_training_metrics.plot_training_loss(
        config_preprocessing.ACDC['saved_model']['unet'] / 'history_unet.json')

    model_training_metrics.plot_training_accuracy(
        config_preprocessing.MM2['saved_model']['unet'] / 'history_unet.json')
    model_training_metrics.plot_training_loss(
        config_preprocessing.MM2['saved_model']['unet'] / 'history_unet.json')

    return None


def figure_additional_resource_3_2():
    """
    Presentation of a pre-processed Oxford Pets image and the corresponding ground truth segmentation.
    (Abyssinian 2).
    :return: None
    """
    path_img = config_preprocessing.OXFORD_PETS['training_set']['path_normalized_images'] / 'Abyssinian_2.npy'
    path_mask = config_preprocessing.OXFORD_PETS['training_set']['path_normalized_masks'] / 'Abyssinian_2.npy'
    img_data = np.load(path_img)
    mask_data = np.load(path_mask)

    fig = plt.figure()
    axes = []
    cols, rows = 2, 1

    axes.append(fig.add_subplot(rows, cols, 1))
    plt.imshow(img_data, interpolation='nearest')
    plt.tick_params(labelsize=7)
    plt.xlabel('px', fontsize=7)
    plt.ylabel('px', fontsize=7)

    axes.append(fig.add_subplot(rows, cols, 2))
    plt.imshow(mask_data, interpolation='nearest')
    plt.tick_params(labelsize=7)
    plt.xlabel('px', fontsize=7)
    plt.ylabel('px', fontsize=7)

    fig.tight_layout()
    plt.show()

    return None


def figure_additional_resource_3_3a():
    """
    Get training curves for Unet on Oxford Pets dataset.
    :return: None
    """
    model_training_metrics.plot_training_accuracy(
        config_preprocessing.OXFORD_PETS['saved_model']['unet'] / 'history_unet.json')
    model_training_metrics.plot_training_loss(
        config_preprocessing.OXFORD_PETS['saved_model']['unet'] / 'history_unet.json')

    return None


def figure_additional_resource_3_3b():
    """
    Get training curves for FCN-8 on Oxford Pets dataset.
    :return: None
    """

    model_training_metrics.plot_training_accuracy(
        config_preprocessing.OXFORD_PETS['saved_model']['fcn_8'] / 'history_fcn_8.json')
    model_training_metrics.plot_training_loss(
        config_preprocessing.OXFORD_PETS['saved_model']['fcn_8'] / 'history_fcn_8.json')

    return None


def figure_additional_resource_3_table_1():
    """
    Get performance metrics values for both UNet and FCN-8 when trained and tested on Oxford Pets.
    :return:
    """
    unet_experiment = testing_unet.predict_on_oxford_pets(1)
    unet_loss, unet_accuracy = testing_unet.test_on_oxford_pets(1)

    fcn8_experiment = testing_fcn8.predict_on_oxford_pets(1)
    fcn8_loss, fcn8_accuracy = testing_fcn8.test_on_oxford_pets(1)

    print('unet experiment on Oxford Pets.....................')
    print('mean dice', round(np.mean(unet_experiment[2]), 3))
    print('mean HD', round(np.mean(unet_experiment[3]), 3))
    print('accuracy: {:5.2f}%'.format(100 * unet_accuracy))
    print('loss: {:5.2f}%'.format(100 * unet_loss))
    print('...................................................')

    print('fcn-8 experiment on Oxford Pets.....................')
    print('mean dice', round(np.mean(fcn8_experiment[2]), 3))
    print('mean HD', round(np.mean(fcn8_experiment[3]), 3))
    print('accuracy: {:5.2f}%'.format(100 * fcn8_accuracy))
    print('loss: {:5.2f}%'.format(100 * fcn8_loss))
    print('...................................................')
