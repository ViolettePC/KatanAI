from visualization import figures

""" ACDC """
from preprocessing.acdc import pipeline as acdc_pipeline
from visualization import acdc_infos

""" OXFORD PETS """
from preprocessing.oxford_pets import pipeline as oxford_pets_pipeline
from visualization import oxford_pets_infos

""" M&Ms-2 """
from preprocessing.mm2 import pipeline as mm2_pipeline
from visualization import mm2_infos

""" UNET """
from u_net import testing as testing_unet
from u_net import training as training_unet

""" FCN-8 """
from fcn_8 import training as training_fcn8


if __name__ == "__main__":
    """ ACDC """
    acdc_pipeline.launch_preprocessing()
    training_unet.train_on_acdc()
    training_fcn8.train_on_acdc()

    """ M&Ms-2 """
    mm2_pipeline.launch_preprocessing()
    training_unet.train_on_mm2()
    training_fcn8.train_on_mm2()

    """ OXFORD PETS """
    oxford_pets_pipeline.launch_preprocessing()
    training_unet.train_on_oxford_pets()
    training_fcn8.train_on_oxford_pets()

    """ Figures """
    figures.figure_1()
    figures.figure_2()
    figures.figure_3()
    figures.figure_6()
    figures.figure_7()
    figures.figure_8()
    figures.figure_9()
