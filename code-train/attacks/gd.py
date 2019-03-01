from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from utils.get_model import get_model 

"""
loss to perform Gradient descent on
"""
def get_gd_loss(y_, y_adv, FLAGS):
  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_adv))
  
  return cross_entropy
  # elif (FLAGS.attack_type == 'new_gd'):
  #   adv_logits = tf.nn.softmax(y_adv)
    
  #   label = tf.argmax(y_, 1)
  #   logit_label = tf.multiply(adv_logits, y_)
    
  #   logit_target = tf.multiply(adv_logits, y_target)
  #   target_loss = tf.reduce_sum(tf.subtract(logit_target, logit_label), 1)
  #   # target_loss = tf.reduce_sum(y_adv)
  #   # target_loss = tf.reduce_mean(tf.reduce_sum(logit_target, 1))
  #   # target_loss = tf.subtract(tf.slice(y_adv, [0, y_target], [-1, 1]), tf.slice(y_adv, [0, label], [-1, 1]))

  #   # return target_loss

  
  # return cross_entropy




"""
Projecting the perturbation on the \epsilon L_infty ball
"""

def project(x_t, x, epsilon):
  
  e = tf.ones(tf.shape(x), tf.float32)*epsilon
  
  proj_x = tf.minimum(x_t, tf.add(x, e))
  proj_x = tf.maximum(proj_x, tf.subtract(x, e))
  return proj_x


"""
Takes as input 
1. x: Input placeholder
2. y_: Labels placeholder (one hot)
3. args["model"]
4. args["num_gd_iter"]
5. args["gd_learning_rate"]
"""

def gd (x, y_, epsilon, FLAGS):
  per = tf.random_uniform(tf.shape(x), minval = -1*epsilon, maxval = epsilon)
  x_t = x + per;  
  # x_t = x
  # Computing the gradient and updating
  for i in range(0, FLAGS.num_gd_iter):

    with tf.variable_scope("model_weights") as scope:
        scope.reuse_variables()
        y_adv = get_model(x_t, FLAGS)
    
    loss = get_gd_loss(y_, y_adv ,  FLAGS)
    grad, = tf.gradients(loss, x_t);

    signed_grad = tf.sign(grad)
    scaled_signed_grad = FLAGS.gd_learning_rate * signed_grad
  
    x_t = (x_t + scaled_signed_grad)
    x_t =  project(x_t, x, epsilon)
    # x_t = tf.clip_by_value(x_t, 0.0, 1.0)
    
  return x_t
    


"""
Takes as input 
1. x: Input placeholder
2. y_: Labels placeholder (one hot)
3. args["model"]
4. args["num_gd_iter"]
5. args["gd_learning_rate"]
"""

def train_gd (x, y_, epsilon, FLAGS):

  per = tf.random_uniform(tf.shape(x), minval = -1*epsilon, maxval = epsilon)
  x_t = x + per;
  
  # x_t = x
  # Computing the gradient and updating
  for i in range(0, FLAGS.num_gd_iter):
    print('updating epsilon', i)
    with tf.variable_scope("model_weights") as scope:
        scope.reuse_variables()
        y_adv = get_model(x_t, FLAGS)
    

    if(FLAGS.loss == 'hinge'):
      H = tf.reduce_max(y_adv * (1 - y_), 1)
      H = tf.reshape(H, [-1, 1])
      H = tf.tile(H, [1, FLAGS.num_classes])
      
      L = tf.nn.relu((1 - y_adv + H) * y_)
      loss = tf.reduce_mean(tf.reduce_max(L, 1))
    
    else: 
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_adv))

    grad, = tf.gradients(loss, x_t);

    signed_grad = tf.sign(grad)
    scaled_signed_grad = FLAGS.gd_learning_rate * signed_grad
  
    x_t = (x_t + scaled_signed_grad)
    x_t =  project(x_t, x, epsilon)
    # x_t = tf.clip_by_value(x_t, 0.0, 1.0)
    
  return x_t
    
