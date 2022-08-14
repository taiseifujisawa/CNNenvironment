# set seed
from keras import backend as K
K.clear_session()
import os
RANDOM_SEED = 1
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
import tensorflow as tf
import numpy as np
import random
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


from pathlib import Path
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl


class CnnConfig:
  def __init__(self):
    # seed
    self.random_seed = 1
    tf.random.set_seed(self.random_seed)
    np.random.seed(self.random_seed)
    random.seed(self.random_seed)

    # file structure
    self.wd = Path.cwd()
    self.datasetdir = self.wd / 'dataset'
    self.splitted_datasetdir = self.wd / 'dataset_splitted'
    self.model_savefile = 'my_model.h5'
    self.load_mode = 'directory'

    # learning
    self.train_test_rate = 0.2
    self.validation_rate = 0.2
    self.optimizer = Adam(learning_rate=0.001)
    self.lossfunc = 'sparse_categorical_crossentropy'
    self.epochs = 100
    self.batchsize = 16
    # callback
    self.earlystopping_patience = 5
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
    }


class Cifar10Config:
  def __init__(self):
    RANDOM_SEED = 1
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    self.train_test_rate = 0.2
    self.validation_rate = 0.2
    self.input_shape = (32, 32)      # numpy, (row, column)
    self.outputs = 10
    self.optimizer = Adam()
    self.lossfunc = 'sparse_categorical_crossentropy'
    self.epochs = 20
    self.batchsize = 128
    self.color = 'rgb'

    self.wd = Path.cwd()
    self.datasetdir = self.wd / 'dataset'
    self.splitted_datasetdir = self.wd / 'dataset_splitted'
    self.model_name = 'my_model'
    self.model_savefile = 'my_model.h5'
    self.last_layername = "last_conv"

    self.earlystopping_patience = 5
    self.reducelearningrate_patience = 2
    self.minimum_learningrate = 0.0001

class Cifar10Config2:
  def __init__(self):
    RANDOM_SEED = 1
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    self.train_test_rate = 0.2
    self.validation_rate = 0.2
    self.input_shape = (32, 32)      # numpy, (row, column)
    self.outputs = 10
    self.optimizer = Adam()
    self.lossfunc = 'categorical_crossentropy'
    self.epochs = 20
    self.batchsize = 128
    self.color = 'rgb'

    self.wd = Path.cwd()
    self.datasetdir = self.wd / 'dataset'
    self.splitted_datasetdir = self.wd / 'dataset_splitted'
    self.model_name = 'my_model'
    self.model_savefile = 'my_model.h5'
    self.last_layername = "last_conv"

    self.earlystopping_patience = 5
    self.reducelearningrate_patience = 2
    self.minimum_learningrate = 0.0001
