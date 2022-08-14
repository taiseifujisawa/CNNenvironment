import tensorflow as tf
import numpy as np
import random
from pathlib import Path
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl


class SignConfig:
  def __init__(self):
    # seed
    self.random_seed = 1

    # file structure
    self.wd = Path.cwd() / 'learning_result'   # every file is saved
    self.datasetdir = Path.cwd() / 'dataset'
    self.splitted_datasetdir = Path.cwd() / 'dataset_splitted'
    self.model_savefile = 'my_model.h5'
    self.load_mode = 'directory'

    # learning
    self.train_test_rate = 0.2
    self.validation_rate = 0.2
    self.optimizer = Adam(learning_rate=0.001)
    self.lossfunc = 'sparse_categorical_crossentropy'
    self.epochs = 20
    self.batchsize = 16
    # callback
    self.earlystopping_patience = 5
    self.reducelearningrate_factor = 0.5
    self.reducelearningrate_patience = 2
    self.minimum_learningrate = 0.0001

    # cnn structure
    self.input_shape = (250, 1000)      # numpy, (row, column)
    self.color = 'grayscale'
    self.model_name = 'my_model'
    self.last_layername = "last_conv"
    self.max_pixelvalue = 255.0

    # data augmentation(do not enter 'validation_split' and 'rescale')
    self.da_cnf = {
      #'rotation_range': 0,
      #'width_shift_range': 0.,
      #'height_shift_range': 0.,
      #'brightness_range': None,
      #'shear_range': 0.,
      #'zoom_range': 0.,
      #'channel_shift_range': 0.,
      #'fill_mode': 'nearest',
      #'cval': 0.,
      #'horizontal_flip': False,
      #'vertical_flip': False,
      #'horizontal_flip': True,
      #'vertical_flip': True,
    }

class Cifar10Config:
  def __init__(self):
    # seed
    self.random_seed = 1

    # file structure
    self.wd = Path.cwd() / 'learning'

    self.datasetdir = Path.cwd() / 'dataset'
    self.splitted_datasetdir = Path.cwd() / 'dataset_splitted'
    self.model_savefile = 'my_model.h5'
    self.load_mode = 'database'

    # learning
    self.train_test_rate = 0.2
    self.validation_rate = 0.2
    self.optimizer = Adam(learning_rate=0.001)
    self.lossfunc = 'sparse_categorical_crossentropy'
    self.epochs = 1
    self.batchsize = 16
    # callback
    self.earlystopping_patience = 5
    self.reducelearningrate_factor = 0.5
    self.reducelearningrate_patience = 2
    self.minimum_learningrate = 0.0001

    # cnn structure
    self.input_shape = (32, 32)      # numpy, (row, column)
    self.color = 'rgb'
    self.model_name = 'my_model'
    self.last_layername = "last_conv"
    self.max_pixelvalue = 255.0

    # data augmentation(do not enter 'validation_split' and 'rescale')
    self.da_cnf = {
      #'rotation_range': 0,
      #'width_shift_range': 0.,
      #'height_shift_range': 0.,
      #'brightness_range': None,
      #'shear_range': 0.,
      #'zoom_range': 0.,
      #'channel_shift_range': 0.,
      #'fill_mode': 'nearest',
      #'cval': 0.,
      #'horizontal_flip': False,
      #'vertical_flip': False,
      'horizontal_flip': True,
      'vertical_flip': True,
    }

class MnistConfig:
  def __init__(self):
    # seed
    self.random_seed = 1

    # file structure
    self.wd = Path.cwd() / 'learning'

    self.datasetdir = Path.cwd() / 'dataset'
    self.splitted_datasetdir = Path.cwd() / 'dataset_splitted'
    self.model_savefile = 'my_model.h5'
    self.load_mode = 'database'

    # learning
    self.train_test_rate = 0.2
    self.validation_rate = 0.2
    self.optimizer = Adam(learning_rate=0.001)
    self.lossfunc = 'sparse_categorical_crossentropy'
    self.epochs = 1
    self.batchsize = 16
    # callback
    self.earlystopping_patience = 5
    self.reducelearningrate_factor = 0.5
    self.reducelearningrate_patience = 2
    self.minimum_learningrate = 0.0001

    # cnn structure
    self.input_shape = (28, 28)      # numpy, (row, column)
    self.color = 'grayscale'
    self.model_name = 'my_model'
    self.last_layername = "last_conv"
    self.max_pixelvalue = 255.0

    # data augmentation(do not enter 'validation_split' and 'rescale')
    self.da_cnf = {
      #'rotation_range': 0,
      #'width_shift_range': 0.,
      #'height_shift_range': 0.,
      #'brightness_range': None,
      #'shear_range': 0.,
      #'zoom_range': 0.,
      #'channel_shift_range': 0.,
      #'fill_mode': 'nearest',
      #'cval': 0.,
      #'horizontal_flip': False,
      #'vertical_flip': False,
      #'horizontal_flip': True,
      #'vertical_flip': True,
    }
