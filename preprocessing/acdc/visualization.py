import nibabel as nib
import matplotlib.pyplot as plt


def all_slices_display(path_img):
    """
    Load the image and display every slices(depth) of it in a figure.
    :param path_img: str
    :return: None
    """
    img_obj = nib.load(path_img)
    img_data = img_obj.get_fdata()
    depth = img_data.shape[2]
    rows = depth % 3 + 3
    cols = 3
    axes = []
    fig = plt.figure()
    for i in range(depth):
        axes.append(fig.add_subplot(rows, cols, i + 1))
        plt.imshow(img_data[:, :, i])
        plt.axis('off')
#       axes[-1].set_title(i)
    fig.tight_layout()
    plt.show()
