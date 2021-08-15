import os
import nibabel as nib
from preprocessing import config_preprocessing
from visualization import nifti
import numpy as np


def nb_of_images():
    """
    Count the number of training images and the number of labeled images.
    :return: int (number of training images), int (number of labeled images)
    """
    patients = os.listdir(config_preprocessing.ACDC['raw_path'])
    img_gt_count = 0
    img_count = 0
    for patient in patients:
        patient_path = config_preprocessing.ACDC['raw_path'] / patient
        contents = os.listdir(patient_path)
        for content in contents:
            if content[-9:] == 'gt.nii.gz':
                img_path = patient_path / content
                depth = nifti.get_dimensions(img_path)[2]
                img_gt_count += depth
            elif content[-9:] == '4d.nii.gz':
                img_path = patient_path / content
                img_obj = nib.load(img_path)
                img_data = img_obj.get_fdata()
                img_count += img_data.shape[3] * img_data.shape[2]

    return img_count, img_gt_count


def check_nb_slices_per_patient():
    """
    Check the number of slices per patient.
    :return: min, max (int) number of slices.
    """
    patients = os.listdir(config_preprocessing.ACDC['raw_path'])
    nb_slices = []
    for patient in patients:
        patient_path = config_preprocessing.ACDC['raw_path'] / patient
        contents = os.listdir(patient_path)
        depths = []
        for content in contents:
            if content == 'Info.cfg' or content[-9:] == '4d.nii.gz':
                continue
            img_path = patient_path / content
            depth = nifti.get_dimensions(img_path)[2]
            depths.append(depth)
        if len(list(set(depths))) != 1:
            print('Issue in the number of slices for : ' + patient)
        else:
            nb_slices.append(depths[0])

    min, max = np.min(nb_slices), np.max(nb_slices)

    return min, max
