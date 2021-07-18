import os
import nibabel as nib
from visualization import nifti


def check_orientation(dir_path):
    patients = os.listdir(dir_path)
    for patient in patients:
        files = os.listdir(dir_path / patient)
        for file in files:
            if file[4:8] == "SA_E":
                img_obj = nib.load(dir_path / patient / file)
                x, y, z = nib.aff2axcodes(img_obj.affine)
                print(x, y, z)

    return 0


def check_img_depth_mm2_raw(path):
    patients = os.listdir(path)
    dimensions = []
    for patient in patients:
        files = os.listdir(path / patient)
        for file in files:
            if file[3:7] == '_SA_':
                dimension = nifti.get_dimensions(path / patient / file)[2]
                dimensions.append(dimension)
    res = list(dict.fromkeys(dimensions))
    res.sort()

    return res
