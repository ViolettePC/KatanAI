from pathlib import Path

KATANAI_ROOT = Path(__file__).parent.parent

ACDC = {
    'id': 0,
    'image_size': (208, 208),
    'raw_path': KATANAI_ROOT / 'data/acdc/raw/training',
    'training_set': {
        'path': KATANAI_ROOT / 'data/acdc/training',
        'path_normalized': KATANAI_ROOT / 'data/acdc/training_normalized',
        'path_normalized_images': KATANAI_ROOT / 'data/acdc/training_normalized' / 'images',
        'path_normalized_masks': KATANAI_ROOT / 'data/acdc/training_normalized' / 'mask',
    },
    'testing_set': {
        'path': KATANAI_ROOT / 'data/acdc/testing',
        'path_normalized': KATANAI_ROOT / 'data/acdc/testing_normalized',
        'path_normalized_images': KATANAI_ROOT / 'data/acdc/testing_normalized' / 'images',
        'path_normalized_masks': KATANAI_ROOT / 'data/acdc/testing_normalized' / 'mask',
    },
    'saved_model': {
        'unet': KATANAI_ROOT / 'saved_model' / 'u_net' / 'acdc'
    }
}

OXFORD_PETS = {
    'id': 1,
    'images_raw_path': KATANAI_ROOT / 'data/oxford_pets' / 'raw' / 'images',
    'masks_raw_path': KATANAI_ROOT / 'data/oxford_pets' / 'raw' / 'trimaps',
    'training_set': {
        'path_images': KATANAI_ROOT / 'data/oxford_pets/training' / 'images',
        'path_masks': KATANAI_ROOT / 'data/oxford_pets/training' / 'masks',
        'path_normalized': KATANAI_ROOT / 'data/oxford_pets/training_normalized',
        'path_normalized_images': KATANAI_ROOT / 'data/oxford_pets/training_normalized' / 'images',
        'path_normalized_masks': KATANAI_ROOT / 'data/oxford_pets/training_normalized' / 'maks'
    },
    'testing_set': {
        'path_images': KATANAI_ROOT / 'data/oxford_pets/testing' / 'images',
        'path_masks': KATANAI_ROOT / 'data/oxford_pets/testing' / 'masks',
        'path_normalized': KATANAI_ROOT / 'data/oxford_pets/testing_normalized',
        'path_normalized_images': KATANAI_ROOT / 'data/oxford_pets/testing_normalized' / 'images',
        'path_normalized_masks': KATANAI_ROOT / 'data/oxford_pets/testing_normalized' / 'maks'
    }
}

MM2 = {
    'training_path': KATANAI_ROOT / 'data/m&m2/training',
    'training_path_normalized': KATANAI_ROOT / 'data/m&m2/training_normalized',
    'testing_path': KATANAI_ROOT / 'data/m&m2/validation'
}
