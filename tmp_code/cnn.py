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
#from keras.utils.image_utils import array_to_img, img_to_array, load_img, save_img


def sign_classifier_cnn(cnf):
  color = (1,) if cnf.color == 'grayscale' else (3,)
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=cnf.input_shape + color),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', name=cnf.last_layername),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(cnf.outputs, activation='softmax')
  ], name=cnf.model_name)
  model.summary()
  model.compile(optimizer=cnf.optimizer, loss=cnf.lossfunc, metrics=['accuracy'])

  return model

def cifar10_cnn(cnf):
  color = (1,) if cnf.color == 'grayscale' else (3,)
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=cnf.input_shape + color),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name=cnf.last_layername),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(cnf.outputs, activation='softmax')
  ], name=cnf.model_name)
  model.summary()
  model.compile(optimizer=cnf.optimizer, loss=cnf.lossfunc, metrics=['accuracy'])

  return model
