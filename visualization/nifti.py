import nibabel as nib
import matplotlib.pyplot as plt


def get_metadata_img(img_path):
    """
    Load header of a Nifti image.
    :param img_path: str
    :return: header object
    """
    img_obj = nib.load(img_path)

    return img_obj.header


def visualize_slice_in_3d_img(img_path, slice_nb):
    """
    Show one slice from a Nifti 3d image
    :param img_path: str
    :param slice_nb: int
    :return: None
    """
    slice = get_slice_in_3d_img(img_path, slice_nb)
    plt.imshow(slice)
    plt.title('slice ' + str(slice_nb))
    plt.show()

    return None


def get_slices_3d_img(img_path):
    """
    Return all slices from a 3D Nitfi image
    :param img_path: str
    :return: array of images
    """
    img_obj = nib.load(img_path)
    img_data = img_obj.get_fdata()
    slices = []
    if len(img_data.shape) == 3:
        depth = img_data.shape[2]
        for i in range(depth):
            slices.append(img_data[:, :, i])

    return slices


def get_slice_in_3d_img(img_path, slice_nb):
    """
    Return one slice from a 3D nifti image
    :param img_path: str
    :param slice_nb: int
    :return: slice image
    """
    img_obj = nib.load(img_path)
    img_data = img_obj.get_fdata()
    if len(img_data.shape) != 3:
        return None
    slice = img_data[:, :, slice_nb]

    return slice


def visualize_slices_3d_img(img_path):
    """
    Show every slices of a 3D nifti image on one figure.
    :param img_path: str
    :return: None
    """
    img_obj = nib.load(img_path)
    img_data = img_obj.get_fdata()
    if len(img_data.shape) == 3:
        depth = img_data.shape[2]
        rows, cols = depth % 3 + 4, 3
        axes = []
        fig = plt.figure()
        if img_data.shape[2] == 1:
            cols, rows = 1, 1
        for i in range(depth):
            axes.append(fig.add_subplot(rows, cols, i + 1))
            plt.imshow(img_data[:, :, i])
            plt.axis('off')
            axes[-1].set_title(i)
        fig.tight_layout()
        plt.show()

    return None


def get_dimensions(img_path):
    """
    Get dimensions of a Nifti image
    :param img_path: str
    :return: list the size of the n dimensions of the image ([height, weight, slices, frames])
    """
    img_obj = nib.load(img_path)
    img_data = img_obj.get_fdata()

    return img_data.shape
