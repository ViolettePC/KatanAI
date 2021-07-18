import os
import random
import shutil


def split_training_testing(config, percentage):
    create_testing_directory(config['testing_set']['path'])
    patients = os.listdir(config['training_set']['path'])
    nb_patient = 0
    for patient in patients:
        nb_patient += 1
    testing_set_count = round(nb_patient - ((nb_patient / 100) * percentage))
    selected_patients = random.sample(os.listdir(config['training_set']['path']), testing_set_count)

    for patient in selected_patients:
            shutil.copytree(config['training_set']['path'] / patient,
                            config['testing_set']['path'] / patient)
            shutil.rmtree(config['training_set']['path'] / patient)

    return None


def create_testing_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    return None
