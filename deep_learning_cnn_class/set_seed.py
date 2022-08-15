def set_seed(seed=1):
  from tensorflow.keras import backend as K
  K.clear_session()
  import os
  os.environ['PYTHONHASHSEED'] = str(seed)
  os.environ['TF_DETERMINISTIC_OPS'] = 'true'
  os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
  import tensorflow as tf
  import numpy as np
  import random
  tf.random.set_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
