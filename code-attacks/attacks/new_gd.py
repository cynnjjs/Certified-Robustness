from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from utils.get_model import get_model 
from train.train_main import get_classification_loss



"""
Projecting the perturbation on the \epsilon L_infty ball
"""

def project(x_t, x, epsilon):
  
  e = tf.ones(tf.shape(x), tf.float32)*epsilon
  
  proj_x = tf.minimum(x_t, tf.add(x, e))
  proj_x = tf.maximum(proj_x, tf.subtract(x, e))
  return proj_x



