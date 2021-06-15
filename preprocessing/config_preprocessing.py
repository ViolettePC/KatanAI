from pathlib import Path

KATANAI_ROOT = Path(__file__).parent.parent

ACDC = {
    'training_path': KATANAI_ROOT / 'data/acdc/training',
    'training_path_normalized': KATANAI_ROOT / 'data/acdc/training_normalized',
    'testing_path': KATANAI_ROOT / 'data/acdc/testing'
}
