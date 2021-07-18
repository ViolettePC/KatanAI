from pathlib import Path

KATANAI_ROOT = Path(__file__).parent.parent

ACDC = {
    'id': 0,
    'image_size': (208, 208),
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
    }
}

OXFORD_PETS = {
    'id': 1,
    'training_set': {
        'path_images': KATANAI_ROOT / 'data/oxford_pets/images',
        'path_masks': KATANAI_ROOT / 'data/oxford_pets/trimaps'
    }
}

MM2 = {
    'training_path': KATANAI_ROOT / 'data/m&m2/training',
    'training_path_normalized': KATANAI_ROOT / 'data/m&m2/training_normalized',
    'testing_path': KATANAI_ROOT / 'data/m&m2/validation'
}
