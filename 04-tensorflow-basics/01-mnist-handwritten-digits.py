"""
Legacy TensorFlow 1.x setup (`tf.contrib.learn`) for MNIST handwritten digit
classification. NOTE: `tf.contrib` was removed in TensorFlow 2.x, so this file
is kept as a historical reference and will not run as-is on modern TensorFlow.
"""

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)
