import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class NormalizationOxfordPets:

    def __init__(self, config, image_size):
        self.image_size = image_size
        self.path_images = config['path_images']
        self.path_masks = config['path_masks']
        self.path_normalized = config['path_normalized']
        self.path_normalized_images = config['path_normalized_images']
        self.path_normalized_masks = config['path_normalized_masks']

    def normalize(self):
        self.create_normalized_directory()
        pets_images = os.listdir(self.path_images)
        for pet in pets_images:
            if pet[-4:] == '.jpg':
                image_path = self.path_images / pet
                numpy_image = self.transform_img_to_numpy_arrays(image_path, 0)
                self.save_numpy_image(numpy_image, 0, pet[:-4])
        pets_masks = os.listdir(self.path_masks)
        for pet in pets_masks:
            if pet[-4:] == '.png':
                mask_path = self.path_masks / pet
                numpy_mask = self.transform_img_to_numpy_arrays(mask_path, 1)
                self.save_numpy_image(numpy_mask, 1, pet[:-4])

        return None

    def create_normalized_directory(self):
        if os.path.exists(self.path_normalized):
            shutil.rmtree(self.path_normalized)
        os.makedirs(self.path_normalized_images)
        os.makedirs(self.path_normalized_masks)

    def regroup_class(self, img):
        img[img > 1] = 0
        img[img == 1] = 2

        return img

    def transform_img_to_numpy_arrays(self, file_path, file_type):

        img = load_img(file_path, target_size=self.image_size, color_mode="grayscale")
        img_array = np.array(img)
        if file_type == 1:
            img_array = self.regroup_class(img_array)

        return img_array

    def save_numpy_image(self, img_normalized, img_type, img_name):
        path = self.path_normalized_images
        if img_type == 1:
            path = self.path_normalized_masks
        np.save(path / img_name, img_normalized, allow_pickle=True)

        return None
