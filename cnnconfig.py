import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import random
from keras.optimizers import Adam, SGD


class CnnConfig:
  def __init__(self):
    RANDOM_SEED = 1
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    self.train_test_rate = 0.2
    self.validation_rate = 0.2
    self.input_shape = (250, 1000)      # numpy, (row, column)
    self.outputs = 2
    self.optimizer = Adam(learning_rate=0.001)
    self.lossfunc = 'sparse_categorical_crossentropy'
    self.epochs = 100
    self.batchsize = 16
    self.color = 'grayscale'

    self.wd = Path.cwd()
    self.datasetdir = self.wd / 'dataset'
    self.splitted_datasetdir = self.wd / 'dataset_splitted'
    self.model_name = 'my_model'
    self.model_savefile = 'my_model.h5'
    self.last_layername = "last_conv"

    self.earlystopping_patience = 5
    self.reducelearningrate_patience = 2
    self.minimum_learningrate = 0.0001


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
