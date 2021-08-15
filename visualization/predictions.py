from matplotlib import pyplot as plt


def show_prediction(testing_set, predictions, dice_values, hausdorff_values, index):
    """
    Create a figure to present the predicted mask, the corresponding ground truth segmentation, and the original image.
    The DICE coefficient and the Hausdorff distance are also added to the figure.
    :param testing_set: testing set object.
    :param predictions: array of predicted masks.
    :param dice_values: array of DICE values.
    :param hausdorff_values: array of Hausdorff distances.
    :param index: int, index of the prediction.
    :return: None.
    """
    fig, axes = plt.subplots(2, 3)

    axes[0][0].imshow(testing_set[0][index])
    axes[0][0].tick_params(labelsize=7)
    axes[0][0].set_title('MRI')

    axes[0][1].imshow(testing_set[1][index])
    axes[0][1].tick_params(labelsize=7)
    axes[0][1].set_title('Ground Truth')

    axes[0][2].imshow(predictions[index])
    axes[0][2].tick_params(labelsize=7)
    axes[0][2].set_title('Predicted Mask')

    axes[1][0].text(0.2, 1, 'DICE Coefficient: {}'.format(round(dice_values[index], 3)))
    axes[1][0].axis('off')

    axes[1][1].text(0.2, 1, 'Hausdorff Distance: {}'.format(round(hausdorff_values[index], 3)))
    axes[1][1].axis('off')

    axes[1][2].axis('off')

    plt.show()

    return None


def show_predictions(testing_set, predictions, i=0, max=10):
    """
    Present a range of predictions compared with their original images and their ground truth segmentations.
    :param testing_set: testing set object.
    :param predictions: array of predicted masks.
    :param i: int, Index of the first prediction to show.
    :param max: int, Maximum number of predictions to show.
    :return: None.
    """
    for _ in predictions:
        fig = plt.figure()
        axes = []
        cols, rows = 3, 1

        axes.append(fig.add_subplot(rows, cols, 1))
        plt.imshow(testing_set[0][i], interpolation='nearest')
        plt.title('image ' + str(i))

        axes.append(fig.add_subplot(rows, cols, 2))
        plt.imshow(testing_set[1][i], interpolation='nearest')
        plt.title('mask ' + str(i))

        axes.append(fig.add_subplot(rows, cols, 3))
        plt.imshow(predictions[i], interpolation='nearest')
        plt.title('prediction ' + str(i))

        fig.tight_layout()
        plt.show()
        i += 1
        if i > i + max:
            break

    return None
