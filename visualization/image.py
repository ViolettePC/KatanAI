import matplotlib.pyplot as plt
import numpy
from visualization import nifti


def display_nifti_slice(path, slice_number):
    patient = str(path).split('/')[-2]
    slice = nifti.get_slice_in_3d_img(path, slice_number)
    plt.imshow(slice)
    plt.title('Raw slice ' + str(slice_number)
              + ' from ' + patient)
    plt.show()


def display_numpy_array_img(path):
    """
    Create a graph of the numpy array image
    :param path: str
    :return: None
    """
    title = str(path).split('/')[-1]
    array = numpy.load(path)
    plt.imshow(array, interpolation='nearest')
    plt.title(title)
    plt.show()

    return None
