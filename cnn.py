from re import U
import tensorflow as tf


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
  model.compile(optimizer=cnf.optimizer, loss=cnf.lossfunc, metrics=['accuracy'])

  return model

def mnist_cnn(cnf):
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
  model.compile(optimizer=cnf.optimizer, loss=cnf.lossfunc, metrics=['accuracy'])

  return model

def cifar10_cnn_vgg16transfer(cnf):
  from tensorflow.keras.applications.vgg16 import VGG16
  from tensorflow.keras.layers import Dense,MaxPooling2D,UpSampling2D,\
    Dropout, Flatten, Input
  from tensorflow.keras import Model
  Vgg16 = VGG16(include_top=False,input_shape=(128,128,3))
  for layer in Vgg16.layers[:15]:
    layer.trainable = False
  _input = Input((32,32,3))
  x = UpSampling2D(_input)
  x = UpSampling2D(x)
  x = Vgg16(x)
  x = Flatten()(x)
  x = Dense(512, activation='relu')(x)
  x = Dropout(0.5)(x)
  output = Dense(10, activation='softmax')(x)
  model = Model(inputs=_input, outputs=output)
  model.compile(optimizer=cnf.optimizer, loss=cnf.lossfunc, metrics=['accuracy'])

  return model
