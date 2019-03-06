"""
File contains model definition of a two layer network 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from models.general_utils import conv2d, max_pool_2x2, weight_variable, bias_variable

"""
Defines the graph for feed forward two layer network 
Takes as input: x (input placeholder) and FLAGS that contains the following
a) num_hidden -- Number of hidden nodes 
b) dimension -- Dimensionality 
c) num_classes -- Number of classes 
"""

def two_layer(x, FLAGS):

    x_ravel = tf.reshape(x, [-1, FLAGS.dimension])
    # First fully connected layer
    """
    W_fc1 = tf.get_variable("W_fc1",   initializer = tf.truncated_normal([FLAGS.dimension, FLAGS.num_hidden], stddev = 0.1))
    b_fc1 = tf.get_variable("b_fc1", initializer=tf.zeros([FLAGS.num_hidden]))
    """
    W_fc1 = weight_variable("W_fc1", [FLAGS.dimension, FLAGS.num_hidden])
    b_fc1 = bias_variable("b_fc1", [FLAGS.num_hidden])
    # ReLU activation
    # Second layer
    W_fc2 = tf.get_variable("W_fc2",   initializer = tf.truncated_normal([FLAGS.num_hidden, FLAGS.num_classes], stddev = 0.1))
    b_fc2 = tf.get_variable("b_fc2", initializer=tf.zeros([FLAGS.num_classes]))
    
    # Here the magic happens
    beta = 5.0
    h_fc1 = tf.nn.softplus(tf.scalar_mul(beta, tf.matmul(x_ravel, W_fc1) + b_fc1))
    y = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y



