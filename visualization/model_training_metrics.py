from matplotlib import pyplot as plt
import json


def plot_training_accuracy(history_path):
    """
    Plot the evolution of the accuracy of both training and validation set during the training.
    :param history_path: training metrics object
    :return: None
    """
    f = open(history_path, )
    data = json.load(f)
    plt.plot(data['accuracy'])
    plt.plot(data['val_accuracy'])
    plt.grid()
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train',
                'validation'], loc='upper left')
    plt.show()


def plot_training_loss(history_path):
    """
    Plot the evolution of the loss of both training and validation set during the training.
    :param history_path: training metrics object
    :return: None
    """
    f = open(history_path, )
    data = json.load(f)

    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.grid()
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train',
                'validation'], loc='upper left')
    plt.show()
