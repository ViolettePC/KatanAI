import os
import shutil
import nibabel as nib
import numpy as np
import cv2


class NormalizationMM2:
    """
    Normalize M&M2 dataset:
    Select only frames of end diastole and end systole view in Short Axis.
    Those frames are being broken down in order to extract every slices of them.
    Those slices are reoriented if necessary and transformed into numpy array.
    The spatial resolution is rescaled to obtain a 1mm^2 / pixel.
    Every numpy array image is saved in a new directory.
    """
    def __init__(self, raw_path, normalized_path):
        """
        :param raw_path: directory containing raw images from M&M2 like downloaded from their website.
        :param normalized_path: temporary directory where normalized images are stored.
        """
        self.raw_path = raw_path
        self.normalized_path = normalized_path

    def normalize(self):
        self.create_normalized_directory()
        patients = os.listdir(self.raw_path)
        for patient in patients:
            files = os.listdir(self.raw_path / patient)
            for file in files:
                if file[3:7] == '_SA_' and file[3:11] != '_SA_CINE':
                    img_path = self.raw_path / patient / file
                    img_arrays = self.transform_img_to_numpy_arrays(img_path)
                    images_reoriented = []
                    pixdim = self.get_spatial_resolution(img_path)
                    for img_array in img_arrays:
                        images_reoriented.append(
                            self.change_orientation(img_array,
                                                    self.get_orientation(img_path)))
                    images_rescaled = self.rescale(images_reoriented, pixdim)
                    self.save_numpy_images(images_rescaled, file[:-7])

        return 0

    def change_orientation(self, img_array, orientation):
        x, y, z = orientation
        if x == 'A' and y == 'R' and z == 'I':

            return cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if x == 'I' and y == 'A' and z == 'R':

            return img_array

        if x == 'I' and y == 'A' and z == 'L':

            return img_array

        elif x == 'I' and y == 'R' and z == 'P':

            return img_array

        elif x == 'L' and y == 'A' and z == 'I':

            return img_array

        elif x == 'A' and y == 'R' and z == 'S':

            return img_array

        elif x == 'I' and y == 'R' and z == 'A':

            return img_array

    def get_orientation(self, img_path):
        img_obj = nib.load(img_path)
        x, y, z = nib.aff2axcodes(img_obj.affine)
        
        return x, y, z

    def save_numpy_images(self, images_normalized, new_name):
        i = 0
        for image in images_normalized:
            n = new_name + '_' + str(i)
            np.save(self.normalized_path / n, image)
            i += 1

        return 0

    def create_normalized_directory(self):
        if os.path.exists(self.normalized_path):
            shutil.rmtree(self.normalized_path)
        os.mkdir(self.normalized_path)

    def transform_img_to_numpy_arrays(self, file_path):
        img_obj = nib.load(file_path)
        img_data = img_obj.get_fdata()
        depth = img_data.shape[2]
        images = []
        for i in range(depth):
            images.append(np.array(img_data[:, :, i]))

        return images

    def get_spatial_resolution(self, file_path):
        img_obj = nib.load(file_path)
        header_infos = img_obj.header

        return header_infos.get_zooms()

    def rescale(self, images, pixdim):
        """
        Rescale every images in order to obtain a spatial resolution of 1mm^2 / pixel
        :param images: array containing numpy arrays of all the slices in a frame
        :param pixdim: spatial resolution of the images
        :return: array
        """
        images_rescaled = []
        for image in images:
            images_rescaled.append(
                cv2.resize(image, (0, 0), fx=pixdim[0], fy=pixdim[1])
            )

        return images_rescaled
