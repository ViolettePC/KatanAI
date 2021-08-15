from preprocessing import config_preprocessing
from u_net.data import DataUNet
from matplotlib import pyplot as plt
import os


def nb_images():
    images = os.listdir(config_preprocessing.OXFORD_PETS['images_raw_path'])

    return len(images)


def show_set_unet(set):
    if set == 'testing':
        set = DataUNet(
            config_preprocessing.OXFORD_PETS['id'],
            config_preprocessing.OXFORD_PETS['testing_set']['path_normalized_images'],
            config_preprocessing.OXFORD_PETS['testing_set']['path_normalized_masks'],
            config_preprocessing.OXFORD_PETS['image_size']
        )[0]
    elif set =='training':
        set = DataUNet(
            config_preprocessing.OXFORD_PETS['id'],
            config_preprocessing.OXFORD_PETS['training_set']['path_normalized_images'],
            config_preprocessing.OXFORD_PETS['training_set']['path_normalized_masks'],
            config_preprocessing.OXFORD_PETS['image_size']
        )[0]
    i = 0
    for img in set[0]:
        fig = plt.figure()
        axes = []
        cols, rows = 2, 1
        axes.append(fig.add_subplot(rows, cols, 1))
        plt.imshow(set[0][i], interpolation='nearest')
        plt.title('image ' + str(i))
        axes.append(fig.add_subplot(rows, cols, 2))
        plt.imshow(set[1][i], interpolation='nearest')
        plt.title('mask ' + str(i))
        fig.tight_layout()
        plt.show()
        i += 1
        if i > 30:
            break