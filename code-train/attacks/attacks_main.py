
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

## Model definition
from utils.get_model import get_model


## Losses
from train.train_new_eig import get_classification_loss

## Attacks 
from attacks.gd import gd 
from attacks.fgsm  import fgsm
from attacks.cw import CarliniLi
"""
Implements the attack on a model 
Takes as input:
1. X_test
2. Y_test
3. attack_type: FGSM or GD or CW
4. epsilon
5. args["fsave"] + "-weights" to restore the weights 
6. args["dimension"], args["num_hidden"], args["num_classes"] for model definition
"""


def attack(X_test, Y_test, attack_type, Epsilon, FLAGS):
    

    ## Placeholders and model 
    x = tf.placeholder(tf.float32, [None, FLAGS.dimension])
    y_= tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    # y_target =tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    with tf.variable_scope("model_weights") as scope:
        y_model = get_model(x, FLAGS)
        scope.reuse_variables()


    ## Running the graph 
        
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ## Restoring the weights
    if(FLAGS.no_train == 0):
        saver.restore(sess, FLAGS.msave + "-weights")
    else :
        saver.restore(sess, FLAGS.msave)

    ##### CW attack here ####
    if(attack_type == 'cw'):
        attack = CarliniLi(sess, FLAGS)
        

        P = np.arange(np.max(np.shape(X_test)))
        np.random.shuffle(P)
        X_test = X_test[P]
        Y_test = Y_test[P]
        

        
        adv = attack.attack(X_test[0:FLAGS.num_test, :], Y_test[0:FLAGS.num_test, :], FLAGS)
        cw_epsilon = np.max( np.abs(adv - X_test[0:FLAGS.num_test, :]), 1)
        
        for e in Epsilon: 
            Zero_one_loss = np.size(np.where(cw_epsilon <= e))/FLAGS.num_test
    
            np.save(os.path.join(FLAGS.results_dir, 'CW-01-' + str(e)),  Zero_one_loss)
        





        return 
    for e in Epsilon: 
    ## Defined the attack tensor
        if(attack_type == 'fgsm'):
            x_attack = fgsm(x, y_, e, FLAGS)
        elif (attack_type == 'gd'):
            x_attack = gd(x, y_ , e, FLAGS)
        with tf.variable_scope("model_weights") as scope:
            scope.reuse_variables()
            y_attack = get_model(x_attack, FLAGS)

    ## Get the losses 
    
        classification_loss = get_classification_loss(y_, y_attack, FLAGS)

        correct_prediction = tf.equal(tf.argmax(y_attack, 1), tf.argmax(y_, 1))
        accuracy = (tf.cast(correct_prediction, tf.float32))
    
        zero_one_loss = 1 - accuracy 
    
    
        Logits_original, = sess.run([y_model], feed_dict={x:X_test, y_:Y_test})
        Logits_new, = sess.run([y_attack], feed_dict={x:X_test, y_:Y_test})
        
        Classification_loss = sess.run([classification_loss], feed_dict={x:X_test, y_:Y_test})


        Zero_one_loss = np.zeros(np.shape(Classification_loss))
        
        for i in range(0, FLAGS.num_gd):            
            New_zero_one_loss = sess.run([zero_one_loss], feed_dict={x:X_test, y_:Y_test})
            Zero_one_loss = np.maximum(New_zero_one_loss, Zero_one_loss)

        
        Accuracy= sess.run([accuracy], feed_dict={x:X_test, y_:Y_test})
        X_new= sess.run([x_attack], feed_dict={x:X_test, y_:Y_test})

        print('Accuracy@' +str(e) , 1 - (np.sum(Zero_one_loss)/np.size(Zero_one_loss)) )

        ## Save it and also return 
        if not os.path.exists(FLAGS.results_dir):
            os.makedirs(FLAGS.results_dir)

        if(attack_type == 'gd'):
            np.save(os.path.join(FLAGS.results_dir, 'GD-01-' + str(e)),  Zero_one_loss)
            np.save(os.path.join(FLAGS.results_dir, 'GD-hinge-' + str(e)), Classification_loss)
            np.save(os.path.join(FLAGS.results_dir, 'Logits_original' + str(e)), Logits_original)
            np.save(os.path.join(FLAGS.results_dir, 'Logits_new_GD' + str(e)), Logits_new)



        if(attack_type == 'fgsm'):
            np.save(os.path.join(FLAGS.results_dir, 'FGSM-01-' + str(e)),  Zero_one_loss)
            np.save(os.path.join(FLAGS.results_dir, 'FGSM--hinge-' + str(e)), Classification_loss)
            np.save(os.path.join(FLAGS.results_dir, 'Logits_new_FGSM' + str(e)), Logits_new)


    return Classification_loss, Zero_one_loss

