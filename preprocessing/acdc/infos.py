import os
from pathlib import Path
import nibabel as nib

TRAINING_PATH = Path(__file__).parent / '../../data/acdc/training/'


def check_number_of_images():
    """
    Verify that every patient in the set got the same number of images and info file.
    :return: 1 if yes, else 0
    """
    patients = os.listdir(TRAINING_PATH)
    num_of_patient = len(patients)
    for patient in patients:
        img, img_gt, info, img_4d = 0, 0, 0, 0
        patient_path = str(TRAINING_PATH) + '/' + patient
        contents = os.listdir(patient_path)
        for content in contents:
            if content == 'Info.cfg':
                info += 1
            elif len(content.split('_')) == 3:
                img_gt += 1
            elif len(content.split('_')) == 2 and content[-9:] == '4d.nii.gz':
                img_4d += 1
            elif len(content.split('_')) == 2 and content[-6:] == 'nii.gz':
                img += 1
        if info == 1 and img_gt == 2 and img == 2 and img_4d == 1:
            continue
        else:
            print('Issue: wrong number files in : ' + patient)
            return 0
    print('... All ' + str(num_of_patient) + ' patients got complete data ... \n'
          '1 info file \n'
          '2 raw images \n'
          '2 annotated images \n'
          '1 4d image')
    return 1


def get_dimensions_nifti(img_path):
    img_obg = nib.load(img_path)
    img_data = img_obg.get_fdata()
    height, weight, depth = img_data.shape

    return height, weight, depth


def check_img_depth_for_every_patients():
    """
    Check the number of slices in every nifti images.
    :return: list containing the different number of slices
    """
    patients = os.listdir(TRAINING_PATH)
    nb_slices = []
    for patient in patients:
        patient_path = str(TRAINING_PATH) + '/' + patient
        contents = os.listdir(patient_path)
        depths = []
        for content in contents:
            if content == 'Info.cfg' or content[-9:] == '4d.nii.gz':
                continue
            img_path = patient_path + '/' + content
            depth = get_dimensions_nifti(img_path)[2]
            depths.append(depth)
        if len(list(set(depths))) != 1:
            print('Issue in the number of slices for : ' + patient)
        else:
            nb_slices.append(depths[0])
    print(list(set(nb_slices)))

    return list(set(nb_slices))
