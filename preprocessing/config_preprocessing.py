from pathlib import Path

KATANAI_ROOT = Path(__file__).parent.parent

ACDC = {
    'id': 0,
    'image_size': (192, 192),
    'raw_path': KATANAI_ROOT / 'data/acdc/raw/training',
    'training_set': {
        'path': KATANAI_ROOT / 'data/acdc/training',
        'path_normalized': KATANAI_ROOT / 'data/acdc/training_normalized',
        'path_normalized_images': KATANAI_ROOT / 'data/acdc/training_normalized' / 'images',
        'path_normalized_masks': KATANAI_ROOT / 'data/acdc/training_normalized' / 'masks'
    },
    'testing_set': {
        'path': KATANAI_ROOT / 'data/acdc/testing',
        'path_normalized': KATANAI_ROOT / 'data/acdc/testing_normalized',
        'path_normalized_images': KATANAI_ROOT / 'data/acdc/testing_normalized' / 'images',
        'path_normalized_masks': KATANAI_ROOT / 'data/acdc/testing_normalized' / 'masks'
    },
    'saved_model': {
        'unet': KATANAI_ROOT / 'saved_model' / 'u_net' / 'acdc',
        'fcn_8': KATANAI_ROOT / 'saved_model' / 'fcn_8' / 'acdc'
    }
}

MM2 = {
    'id': 2,
    'image_size': (192, 192),
    'raw_path': KATANAI_ROOT / 'data/mm2' / 'raw' / 'training',
    'training_set': {
        'path': KATANAI_ROOT / 'data/mm2/training',
        'path_normalized': KATANAI_ROOT / 'data/mm2/training_normalized',
        'path_normalized_images': KATANAI_ROOT / 'data/mm2/training_normalized' / 'images',
        'path_normalized_masks': KATANAI_ROOT / 'data/mm2/training_normalized' / 'masks',
    },
    'testing_set': {
        'path': KATANAI_ROOT / 'data/mm2/testing',
        'path_normalized': KATANAI_ROOT / 'data/mm2/testing_normalized',
        'path_normalized_images': KATANAI_ROOT / 'data/mm2/testing_normalized' / 'images',
        'path_normalized_masks': KATANAI_ROOT / 'data/mm2/testing_normalized' / 'masks',
    },
    'saved_model': {
        'unet': KATANAI_ROOT / 'saved_model' / 'u_net' / 'mm2',
        'fcn_8': KATANAI_ROOT / 'saved_model' / 'fcn_8' / 'mm2'
    }
}

OXFORD_PETS = {
    'id': 1,
    'image_size': (160, 160),
    'images_raw_path': KATANAI_ROOT / 'data/oxford_pets' / 'raw' / 'images',
    'masks_raw_path': KATANAI_ROOT / 'data/oxford_pets' / 'raw' / 'trimaps',
    'training_set': {
        'path_images': KATANAI_ROOT / 'data/oxford_pets/training' / 'images',
        'path_masks': KATANAI_ROOT / 'data/oxford_pets/training' / 'masks',
        'path_normalized': KATANAI_ROOT / 'data/oxford_pets/training_normalized',
        'path_normalized_images': KATANAI_ROOT / 'data/oxford_pets/training_normalized' / 'images',
        'path_normalized_masks': KATANAI_ROOT / 'data/oxford_pets/training_normalized' / 'masks'
    },
    'testing_set': {
        'path_images': KATANAI_ROOT / 'data/oxford_pets/testing' / 'images',
        'path_masks': KATANAI_ROOT / 'data/oxford_pets/testing' / 'masks',
        'path_normalized': KATANAI_ROOT / 'data/oxford_pets/testing_normalized',
        'path_normalized_images': KATANAI_ROOT / 'data/oxford_pets/testing_normalized' / 'images',
        'path_normalized_masks': KATANAI_ROOT / 'data/oxford_pets/testing_normalized' / 'masks'
    },
    'saved_model': {
        'unet': KATANAI_ROOT / 'saved_model' / 'u_net' / 'oxford_pets',
        'fcn_8': KATANAI_ROOT / 'saved_model' / 'fcn_8' / 'oxford_pets'
    }
}
