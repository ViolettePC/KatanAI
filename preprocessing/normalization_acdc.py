import os
import nibabel as nib
import cv2
import numpy as np
import shutil


class NormalizationACDC:
    """
    rescale images.
    """
    def __init__(self, raw_path, normalized_path):
        self.raw_path = raw_path
        self.normalized_path = normalized_path

    def normalize(self):
        self.create_normalized_directory()
        patients = os.listdir(self.raw_path)
        for patient in patients:
            files = os.listdir(self.raw_path / patient)
            for file in files:
                file_path = self.raw_path / patient / file
                if self.get_file_type(file) == 0 or \
                        self.get_file_type(file) == 1:
                    images = self.transform_img_to_numpy_arrays(file_path)
                    pixdim = self.get_spatial_resolution(file_path)
                    images_rescaled = self.rescale(images, pixdim)
                    new_name = file[:-7]
                    self.save_numpy_images(images_rescaled, new_name)

        return None

    def create_normalized_directory(self):
        if os.path.exists(self.normalized_path):
            shutil.rmtree(self.normalized_path)
        os.mkdir(self.normalized_path)

        return None

    def get_file_type(self, file_name):
        """
        Label the type of file. 4 options possible.
        Type 0 = unlabeled ES and ED frames
        Type 1 = labeled ES and ED frames
        Type 2 = Complete 4D image
        Type 3 = Info file
        :param file_name:
        :return:
        """
        if file_name[-9:] == "gt.nii.gz":
            return 1
        elif file_name[-9:] == "4d.nii.gz":
            return 2
        elif file_name == 'Info.cfg':
            return 3
        else:
            return 0

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
        images_rescaled = []
        for image in images:
            images_rescaled.append(
                cv2.resize(image, (0, 0), fx=pixdim[0], fy=pixdim[1])
            )

        return images_rescaled

    def save_numpy_images(self, images_normalized, new_name):
        i = 0
        for image in images_normalized:
            n = new_name + '_' + str(i)
            np.save(self.normalized_path / n, image)
            i += 1

        return 0
