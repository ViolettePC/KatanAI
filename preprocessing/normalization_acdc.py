import os
import nibabel as nib
import cv2
import numpy as np
import shutil


class NormalizationACDC:
    """
    Normalize ACDC dataset:
    Select frames of end systole and end diastole in short axis view.
    Those frames are broken down to extract every slices of them.
    Every slices are transformed into numpy array.
    Those numpy images are rescaled to obtain a spatial resolution of 1mm*2 / pixel.
    Finally those images are stored into a temporary directory.
    """
    def __init__(self, config):
        self.path = config['path']
        self.path_normalized = config['path_normalized']
        self.path_normalized_images = config['path_normalized_images']
        self.path_normalized_masks = config['path_normalized_masks']

    def normalize(self):
        self.create_normalized_directory()
        patients = os.listdir(self.path)
        for patient in patients:
            files = os.listdir(self.path / patient)
            for file in files:
                file_path = self.path / patient / file
                new_name = file[:-7]
                if self.get_file_type(file) == 0:
                    images = self.transform_img_to_numpy_arrays(file_path)
                    pixdim = self.get_spatial_resolution(file_path)
                    images_rescaled = self.rescale(images, pixdim)
                    images_croped = self.crop(images_rescaled)
                    images_clahe = self.clahe(images_croped)
                    self.save_numpy_images(images_clahe, new_name, 0)

                elif self.get_file_type(file) == 1:
                    images = self.transform_img_to_numpy_arrays(file_path)
                    pixdim = self.get_spatial_resolution(file_path)
                    images_rescaled = self.rescale(images, pixdim)
                    images_croped = self.crop(images_rescaled)
                    self.save_numpy_images(images_croped, new_name, 1)

        return None

    def clahe(self, img_arrays, tile_size=(1, 1)):
        """
        Apply the contrast limited adaptive histogram equalization to every slices of a frame.
        :param img_arrays: numpy arrays of slices of a frame
        :param tile_size: size of the tiles
        :return: array containing images with CLAHE transformation.
        """
        images_clahe = []
        clahe = cv2.createCLAHE(tileGridSize=tile_size)
        for img in img_arrays:
            img_8bit = self.convert_16bit_8bit(img)
            images_clahe.append(clahe.apply(img_8bit))

        return images_clahe

    def convert_16bit_8bit(self, img):
        """
        Convert a 16bit image to a 8bit image
        :param img: numpy array of the 16 bit image
        :return: numpy image of the 8 bit image
        """
        min_16 = np.min(img)
        max_16 = np.max(img)
        img_8 = np.array(np.rint((255 * (img - min_16)) / float(max_16 - min_16)), dtype=np.uint8)

        return img_8

    def crop(self, img_arrays, crop_size=208):
        """
        Crop the image around center.
        :param img_arrays: arrays containing every slices of one frame.
        :param crop_size: wanted size for the newly cropped images (square shape)
        :return: array containing newly cropped images for every slices of a frame.
        """
        images_croped = []
        for img in img_arrays:
            height, width = img.shape
            crop_x = (width - crop_size) // 2
            crop_y = (height - crop_size) // 2
            image_croped = img[crop_y:(crop_y + crop_size), crop_x:(crop_x + crop_size)]
            images_croped.append(image_croped)

        return images_croped

    def create_normalized_directory(self):
        if os.path.exists(self.path_normalized):
            shutil.rmtree(self.path_normalized)
        os.mkdir(self.path_normalized)
        os.mkdir(self.path_normalized_images)
        os.mkdir(self.path_normalized_masks)

        return None

    def get_file_type(self, file_name):
        """
        Label the type of file. 4 options possible.
        Type 0 = unlabeled ES and ED frames
        Type 1 = labeled ES and ED frames
        Type 2 = Complete 4D image
        Type 3 = Info file
        :param file_name:
        :return: int
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

    def save_numpy_images(self, images_normalized, new_name, img_type):
        if img_type == 1:
            path = self.path_normalized_masks
        else:
            path = self.path_normalized_images
        i = 0
        for image in images_normalized:
            n = new_name + '_' + str(i)
            np.save(path / n, image)
            i += 1

        return 0
