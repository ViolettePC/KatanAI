import os
import numpy as np


class DataFCN8:
    """
    Present images/masks to model.
    """
    def __init__(self, dataset_id, normalized_path_images, normalized_path_masks, image_size):
        self.dataset_id = dataset_id
        self.normalized_path_images = normalized_path_images
        self.normalized_path_masks = normalized_path_masks
        self.image_size = image_size

    def join_data(self):
        input_img_paths = sorted(
            [
                os.path.join(self.normalized_path_images, file)
                for file in os.listdir(self.normalized_path_images)
            ]
        )
        target_img_paths = sorted(
            [
                os.path.join(self.normalized_path_masks, file)
                for file in os.listdir(self.normalized_path_masks)
            ]
        )
        batch_size = len(input_img_paths)

        return input_img_paths, target_img_paths, batch_size

    def __getitem__(self, idx):
        input_img_paths, target_img_paths, batch_size = self.join_data()
        i = idx * batch_size
        batch_input = input_img_paths[i: i + batch_size]
        batch_target = target_img_paths[i: i + batch_size]
        x = np.zeros((batch_size,) + self.image_size, dtype="float32")
        names_images = []
        for j, path in enumerate(batch_input):
            img = np.load(path, allow_pickle=True)
            names_images.append(path.split('/')[-1])
            x[j] = img
        y = np.zeros((batch_size,) + self.image_size, dtype="float32")
        names_masks = []
        for j, path in enumerate(batch_target):
            img = np.load(path, allow_pickle=True)
            names_masks.append(path.split('/')[-1])
            y[j] = img
        return x, y, names_images, names_masks
