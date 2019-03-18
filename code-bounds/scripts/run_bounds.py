import tensorflow as tf
import numpy as np
import sys
import os 
import json


# dataset 
from utils.load_mnist import load_mnist

# bounds 
from bounds.bounds_main import bounds_main
from bounds.bounds_main_third import bounds_main_finer

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'mnist', 'dataset')
tf.app.flags.DEFINE_integer('num_classes', 2, 'number of classes')
tf.app.flags.DEFINE_string('model', 'two_layer', 'model: two_layer or multi_layer')

tf.app.flags.DEFINE_integer('num_hidden', 500, 'number of hidden nodes')
tf.app.flags.DEFINE_integer('dimension', 784, 'Dimension of the dataset')
tf.app.flags.DEFINE_integer('num_layers', 2, 'number of layers in network')


tf.app.flags.DEFINE_string('msave', 'weights/temp', 'path to save the model weights')
tf.app.flags.DEFINE_string('results_dir', 'results/temp', 'directory to save the losses')

tf.app.flags.DEFINE_string('reg_type', 'None', 'type of regularization: None, first_order, fro, spectral')
tf.app.flags.DEFINE_float('reg_param', 0.01, 'regularization parameter')
tf.app.flags.DEFINE_integer('num_epochs', 50, 'number of epochs')
tf.app.flags.DEFINE_integer('batch_size', 100, 'batch size')
tf.app.flags.DEFINE_float('alpha', 1e-3, 'alpha')
tf.app.flags.DEFINE_float('beta1', 0.9, 'beta 1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'beta 2')
tf.app.flags.DEFINE_integer('num_gd_iter', 40, 'number of gradient descent iterations')
tf.app.flags.DEFINE_float('gd_learning_rate', 0.01, 'learning rate of gd')
tf.app.flags.DEFINE_string('class_path', 'None', 'path for classes')
tf.app.flags.DEFINE_integer('start', 0, 'start index of test images')
tf.app.flags.DEFINE_integer('end', 1, 'end index of test images')
tf.app.flags.DEFINE_float('noise_param', 1e-6, 'parameter for noising before eigen decomposition')
tf.app.flags.DEFINE_integer('data', 0, 'set 1 for data dependent experiments')
tf.app.flags.DEFINE_integer('num_gd', 1, 'number of restarts of gradient descent')
tf.app.flags.DEFINE_float('train_epsilon', 0.3, 'epsilon to train for')
tf.app.flags.DEFINE_float('test_epsilon', 0.1, 'epsilon to test for')
tf.app.flags.DEFINE_integer('change_epoch', -1 , 'Epoch to change regularization parameter')
tf.app.flags.DEFINE_float('new_reg_param', 0.0 , 'New regularization parameter')
tf.app.flags.DEFINE_string('checkPoint', None , 'Restoring checkpoint')
tf.app.flags.DEFINE_integer('restore_epoch', 0 , 'Restoring epoch')
tf.app.flags.DEFINE_string('loss', 'cross-entropy' , 'Loss to use for training')
tf.app.flags.DEFINE_string('gpu', '0' , 'Gpus to use')
tf.app.flags.DEFINE_float('adam_epsilon', 1E-6 , 'Adam parameter epsilon')
tf.app.flags.DEFINE_float('sd', 0.5 , 'Standard deviation of initialization')
tf.app.flags.DEFINE_string('opt', 'adam' , 'Optimizer')
tf.app.flags.DEFINE_float('scale_dual', 1 , 'Scale for dual variables')
tf.app.flags.DEFINE_integer('weights_change', 600, 'weights change every these many epochs')
tf.app.flags.DEFINE_integer('weights_option', 0, 'One of four weight options')
tf.app.flags.DEFINE_float('normal_param', 1, 'Weighting of original classification loss')
tf.app.flags.DEFINE_integer('run_cw', 0, 'Whether to run Carlini Wagner attack')
tf.app.flags.DEFINE_integer('run_gd', 0, 'Whether to run gradient descent attack ')
tf.app.flags.DEFINE_integer('run_fgsm', 0, 'Whether to run fast gradient sign method')
tf.app.flags.DEFINE_integer('num_test', 1000, 'Number of points to run the CW attack on')
tf.app.flags.DEFINE_integer('no_train', 0, 'When set to 1, there is no training')
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

"""
script that computes all the losses (attacks and bounds) 
take the following as input
1. dataset
2. model
3. number of hidden nodes
4. dimensionality of dataset 
5. msave: where the trained models are being saved/saved already 
6. results_dir: directory to save all the losses (loss is a vector for all datapoints indexed by epsilon
7. regularization type
8. regularization parameter

(all the above are flags; there are other parameters like learning rate, number of iterations etc. as well)
todo: make changes to bounds/attacks to get min_epsilon
"""

def main(argv=None):


    if (FLAGS.dataset  == 'mnist'):
        FLAGS.dimension = 784
        X_train, Y_train, X_test, Y_test = load_mnist(FLAGS)

    else:
        print("Invalid dataset")
            


#Epsilon = np.linspace(0, 0.2, 10)
    Epsilon = [0.1]
    bounds_main_finer(X_test, Y_test,  Epsilon, FLAGS)
    
    

    

if __name__ == '__main__':
    tf.app.run()









