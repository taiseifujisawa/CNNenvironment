import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math
import random
import cv2
from PIL import Image
import pickle
from tqdm import tqdm
import traceback
import shutil
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import array_to_img, img_to_array, load_img, save_img


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
