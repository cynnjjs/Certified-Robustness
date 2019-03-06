import tensorflow as tf
import numpy as np
import cvxpy as cp
import os
import sys

from utils.get_model import get_model

## Bounds
from bounds.bounds_sdp import get_pairwise_loss
from bounds.bounds_spectral import bounds_spectral 
from bounds.bounds_fro import bounds_fro
from bounds.bounds_so import qp_solver, qp_feasibility, sdp_solver, eq_solver

"""
Computes the loss for each bound
Takes as input: 
1. X_test --> To compute the current function values 
2. Y_test
3. FLAGS for defining the model 
"""

# Calculate sigma' for softplus
def sigma_prime (t, beta):
    return beta / (1.0+np.exp(-beta*t))

# Calculate gradient at x
def grad_2 (x, a, beta):
    return (beta**2) * np.exp(-beta*(a+x)) / (x * (1.0 + np.exp(-beta*(a+x)))**2) - (sigma_prime(x+a, beta)-sigma_prime(a, beta)) / (x**2)

# Hacky upper bound, x < 0
def hacky_ub (x):
    # For beta = 5 only
    a, b, c, d, e = [1.27167784, -0.15555676, 0.46238266, 5.64449646, 1.55300805]
    return a/(x*x-b*x+c) + d/(-x+e)

def compute_logits (X_test, Y_test, FLAGS):
## Step 1: Define the model 
## Step 2: Load the weights
## Step 3: Run the session to get the logits 

    x = tf.placeholder(tf.float32, [None, FLAGS.dimension])
    with tf.variable_scope("model_weights") as scope:
        scope.reuse_variables()
        y_model = get_model(x, FLAGS)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ## Restoring the weights and dual variables 
    saver.restore(sess, FLAGS.msave + "-final")

    ## Running the graph 
    Logits, = sess.run([y_model], feed_dict={x:X_test})

    return Logits

"""
Takes as input:
 1. Logits
2. Bounds_matrix 
Returns the zero one and hinge loss 
"""
def compute_loss(Y_test, Logits, Bounds_matrix, epsilon, FLAGS):
## For every datapoint, compute hinge loss for max
    num_points = np.shape(Logits)[0]
    num_classes = FLAGS.num_classes
    Zo_loss = np.zeros([num_points, 1])
    Hinge_loss = np.zeros([num_points, 1])
    New_labels = np.zeros([num_points, 1])

    for i in range(num_points):
        ## Compute the label 
        label = np.argmax(Y_test[i, :])
        ## Compute max f(j) - f(true) after perturbation

        new_values = np.ravel(Logits[i, :]) - np.ravel(np.ones([num_classes, 1])*Logits[i, label]) + epsilon*np.ravel(Bounds_matrix[label, :])
        
        max_new_val = np.max(new_values)
        new_label = np.argmax(new_values)
        New_labels[i] = new_label

        if(max_new_val > 0):
            Zo_loss[i] = 1
        
        Hinge_loss[i] = max(0, 1 + max_new_val)
    
    return Zo_loss, Hinge_loss, New_labels

"""
    Have to modify compute_loss because we calculate the bound-matrix based on X_test.
    Takes as input:
    1.
    Returns the zero one and hinge loss
    """
def compute_loss_so(model_weights, X_test, Y_test, Logits, epsilon, FLAGS):
    
    # set beta here
    beta = 5.0
    
    ## For every datapoint, compute hinge loss for max
    num_points = np.shape(Logits)[0]
    num_classes = FLAGS.num_classes
    Zo_loss = np.zeros([num_points, 1])
    Hinge_loss = np.zeros([num_points, 1])
    New_labels = np.zeros([num_points, 1])
    
    W_FC1 = model_weights["W_FC1"]
    # Dimension is [num_hidden, classes]
    W_FC2 = model_weights["W_FC2"]
    # Dimension is [num_classes, num_hidden]
    B_FC1 = model_weights["B_FC1"]
    B_FC2 = model_weights["B_FC2"]
    
    # Preliminary work for so term
    W_i2 = np.zeros((FLAGS.num_hidden, FLAGS.dimension, FLAGS.dimension))
    b_0 = np.zeros((FLAGS.num_hidden))
    
    for j in range(FLAGS.num_hidden):
        # Calculate W_j W_j^T
        W_i2[j] = np.outer(W_FC1[:, j], W_FC1[:, j])
        # Calculate l1 norm of each row of W_1
        b_0[j] = np.linalg.norm(W_FC1[:, j], 1)
    b_0 = b_0 * epsilon

    a_pos = np.zeros((num_classes, num_classes, FLAGS.num_hidden))
    a_neg = np.zeros((num_classes, num_classes, FLAGS.num_hidden))
    for label in range(num_classes):
        for k in range(num_classes):
            if (k != label):
                temp_a = W_FC2[k]-W_FC2[label]
                a_pos[label, k] = np.clip(temp_a, 0, np.max(temp_a))
                a_neg[label, k] = np.clip(temp_a, np.min(temp_a), 0)

    for i in range(num_points):
        # get test data x
        x_ravel = tf.reshape(X_test[i, :], [-1, FLAGS.dimension])
        pre_act = np.ravel((tf.matmul(x_ravel, W_FC1) + B_FC1).eval())
        
        # Compute gradient of f(x)
        sigma_p = beta * tf.nn.sigmoid(beta * pre_act)
        sigma_p_ravel = tf.reshape(sigma_p, [-1]).eval()

        # Gradient of ReLU below in comparison
        """
        sigma_prime_ravel = np.zeros(FLAGS.num_hidden, dtype=np.float32)
        for j in range(FLAGS.num_hidden):
            if pre_act[j] > 0:
                sigma_prime_ravel[j] = 1.0
        """
        grad_fx = tf.matmul(tf.matmul(W_FC2, np.diag(sigma_p_ravel)),np.transpose(W_FC1)).eval()
        
        ## Compute the label
        label = np.argmax(Y_test[i, :])
        
        # First-order approximation
        """
        grad_l1 = np.zeros(num_classes)
        for j in range(num_classes):
            grad_l1[j] = np.linalg.norm(grad_fx[j,:]-grad_fx[label,:], 1)
        """
        # Second-order term
        # so_term = np.zeros(num_classes)
        pre_act_neg = -np.abs(pre_act)
        grad_b0 = np.sign(grad_2(b_0, pre_act_neg, beta)) # Indicator
        grad_b0 = np.clip(grad_b0, 0, np.max(grad_b0))
        
        # sig_pri_2_max: FLAGS.num_hidden

        # Max go from pre_act_neg to opt unless grad_b0 > 0
        candidate_1 = (sigma_prime(pre_act_neg + b_0, beta) - sigma_prime(pre_act_neg, beta)) / b_0
        candidate_2 = hacky_ub(pre_act_neg)
        sig_pri_2_max = candidate_1 * grad_b0 + candidate_2 * (1 - grad_b0)
        
        # Min go from pre_act - b_0 to pre_act
        sig_pri_2_min = (sigma_prime(pre_act_neg, beta) - sigma_prime(pre_act_neg - b_0, beta)) / b_0

        for k in range(num_classes):
            
            psd_M = np.einsum('i, i, ijk -> jk', sig_pri_2_max, a_pos[label, k], W_i2) + np.einsum('i, i, ijk -> jk', sig_pri_2_min, a_neg[label, k], W_i2)
            
            # CVXpy solver for SDP relaxation
            """
            if (k != label):
                print('Solving SDP for class', k)
                so_term[k] = sdp_solver(psd_M, FLAGS.dimension)
            """
            # CVXpy solver for SDP feasibility
            """
            if (k != label):
                print('Solving SDP feasibility for Example', i, 'class', k)
                sys.stdout.flush()
                margin = (-Logits[i, k]+Logits[i, label] - grad_l1[k]*epsilon)/epsilon/epsilon*2
                sdp_status = sdp_feasibility(psd_M, FLAGS.dimension, margin)
                if sdp_status in ["optimal", "unbounded"]:
                    Zo_loss[i] = 1
                    print('Loss at Example',i, 'Class',k)
                    # May not be the new max - just the first class that surpasses true label
                    New_labels[i] = k
                    break
            """
            # CVXpy solver for QP feasibility
            """
            if (k != label):
                print('Solving QP feasibility for Example', i, 'class', k)
                sys.stdout.flush()
                qp_status = qp_feasibility(psd_M, FLAGS.dimension, epsilon, np.ravel(grad_fx[k,:]-grad_fx[label,:]), -Logits[i, k]+Logits[i, label])
                if qp_status in ["optimal", "unbounded"]:
                    Zo_loss[i] = 1
                    print('Loss at Example',i, 'Class',k)
                    # May not be the new max - just the first class that surpasses true label
                    New_labels[i] = k
                    break
            """
            # CVXpy solver for QP
            
            if (k != label):
                print('Solving QP - SDP relax for Example', i, 'class', k)
                sys.stdout.flush()
                fo_term, so_term = qp_solver(psd_M, FLAGS.dimension, epsilon, np.ravel(grad_fx[k,:]-grad_fx[label,:]))
                new_val = Logits[i, k]-Logits[i, label] + fo_term + so_term
                print('Example',i, 'Class',k, Logits[i, k]-Logits[i, label], fo_term, so_term)
                if new_val > 0:
                    Zo_loss[i] = 1
                    print('Loss at Example',i, 'Class',k)
                    # May not be the new max - just the first class that surpasses true label
                    New_labels[i] = k
                    break
            
            """
            new_val = Logits[i, k]-Logits[i, label] + grad_l1[k]*epsilon + so_term[k]*epsilon*epsilon*0.5
            print('Example',i, 'Class',k, Logits[i, k]-Logits[i, label], grad_l1[k]*epsilon, so_term[k]*epsilon*epsilon*0.5)
            if (new_val > 0):
                Zo_loss[i] = 1
                # May not be the new max - just the first class that surpasses true label
                New_labels[i] = k
                break
            """
        if i % 5 == 0:
            print('zo-loss at test example 0 to ',i,':', np.sum(Zo_loss)/(i+1))
        
        ## Compute max f(j) - f(true) after perturbation
        """
        new_values = np.ravel(Logits[i, :]) - np.ravel(np.ones([num_classes, 1]) * Logits[i, label]) + epsilon * (grad_l1 + so_term * 0.5 * epsilon)
        
        max_new_val = np.max(new_values)
        new_label = np.argmax(new_values)
        New_labels[i] = new_label
        
        if(max_new_val > 0):
            Zo_loss[i] = 1
            print('loss at example', i)

        if i % 20 == 0:
            print('zo-loss at test example 0 to ',i,':', np.sum(Zo_loss)/(i+1))
        
        Hinge_loss[i] = max(0, 1 + max_new_val)
        """
    return Zo_loss, Hinge_loss, New_labels

""" 
Restore the weights, and then save the weights
separately as a dictionary (two_layer)
and call the appropriate two_layer bounds 
Takes as input: 
a. X_test 
b. Y_test 
c. args["fsave"] + weights that has the weights
d. args["model"] : Model to build
e. args["results_dir"]: Directory inside which to add the loss files 
f. args["Epsilon"]: Range of epsilon to compute bounds over
"""

def bounds_main_finer(X_test, Y_test, Epsilon, FLAGS):
    x = tf.placeholder(tf.float32, [None, FLAGS.dimension])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    with tf.variable_scope("model_weights") as scope:
        y_model = get_model(x, FLAGS)
        scope.reuse_variables()

    # If reg_type = first order, create the dual variables 
    if(FLAGS.reg_type == "first_order"):
        print("inside first order")
        with tf.variable_scope("dual") as scope:
            num_dual_variables = int( FLAGS.num_classes*(FLAGS.num_classes - 1)*0.5)
            init = np.random.normal(0, FLAGS.sd, [num_dual_variables, FLAGS.num_hidden + FLAGS.dimension + 1]);
            init = np.float32(init)
            c = tf.get_variable("dual", dtype = tf.float32, initializer = init)

    tvars = tf.trainable_variables()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ## Restoring the weights
    

    saver.restore(sess, FLAGS.msave + "-final")
    
    w_fc1  = tvars[0]
    b_fc1 = tvars[1]
    w_fc2 = tvars[2]
    b_fc2 = tvars[3]
    W_fc1 = w_fc1.eval()
    B_fc1 = b_fc1.eval()
    W_fc2 = w_fc2.eval()
    B_fc2 = b_fc2.eval()        


    # Checking if dual variables are present
    if(FLAGS.reg_type == "first_order"):
        c = tvars[4];
        C = c.eval();
        model_weights= {'W_FC1':(W_fc1), 'B_FC1':B_fc1, 'W_FC2':np.transpose(W_fc2), 'B_FC2':B_fc2, 'dual':C}        
        
        SDP_matrix = np.zeros([FLAGS.num_classes, FLAGS.num_classes])
        for i_ in range(FLAGS.num_classes):
            for j_ in range(i_):
                ind = i_*(FLAGS.num_classes - 1) + j_
                SDP_matrix[i_, j_] = get_pairwise_loss(model_weights, i_, j_, FLAGS)
            
        # Scaling factor from the dual formulation 
        SDP_matrix = SDP_matrix*0.25;


    ## Created the dictionary of required weight matrices
    else:
        model_weights= {'W_FC1':(W_fc1), 'B_FC1':B_fc1, 'W_FC2':np.transpose(W_fc2), 'B_FC2':B_fc2}
    
    
    Spectral_matrix = bounds_spectral(model_weights, FLAGS)
    Fro_matrix = bounds_fro(model_weights, FLAGS)
    
    ## Returns the logits as numpy files
    Logits= compute_logits(X_test, Y_test, FLAGS)

    # os.mkdir(FLAGS.results_dir)

    # ## Compute  and save for range of epsilon 
    for e in Epsilon:
        if(FLAGS.reg_type == "first_order"):
            Sdp_loss_01, Sdp_loss_hinge, Sdp_new_labels = compute_loss(Y_test, Logits, SDP_matrix, e, FLAGS)
            print('SDP loss at epsilon = ',e,':', np.sum(Sdp_loss_01)/np.size(Sdp_loss_01))
        Spectral_loss_01, Spectral_loss_hinge, Spectral_new_labels = compute_loss(Y_test, Logits, Spectral_matrix, e, FLAGS)
        Fro_loss_01, Fro_loss_hinge, Fro_new_labels = compute_loss(Y_test, Logits, Fro_matrix, e, FLAGS)

    # Compute Second-order Loss
        if(FLAGS.reg_type == "second_order"):
            So_loss_01, So_loss_hinge, So_new_labels = compute_loss_so(model_weights, X_test, Y_test, Logits, e, FLAGS)
            print('Second_order loss at epsilon = ',e,':', np.sum(So_loss_01)/np.size(So_loss_01))

        ## Save here (indexed by epsilon)

    """
        if(FLAGS.reg_type == "first_order"):
            np.save(os.path.join(FLAGS.results_dir, 'SDP-01-' + str(e)),  Sdp_loss_01)
            np.save(os.path.join(FLAGS.results_dir, 'SDP-hinge-' + str(e)), Sdp_loss_hinge)
            np.save(os.path.join(FLAGS.results_dir, 'SDP-new-labels' + str(e)), Sdp_new_labels)

        np.save(os.path.join(FLAGS.results_dir, 'Spectral-01-' + str(e)), Spectral_loss_01)
        np.save(os.path.join(FLAGS.results_dir, 'Spectral-hinge-' + str(e)), Spectral_loss_hinge)
        np.save(os.path.join(FLAGS.results_dir, 'Spectral-new-labels' + str(e)), Spectral_new_labels)

 
        np.save(os.path.join(FLAGS.results_dir, 'Fro-01-' + str(e)), Fro_loss_01)
        np.save(os.path.join(FLAGS.results_dir, 'Fro-hinge-' + str(e)), Fro_loss_hinge)
        np.save(os.path.join(FLAGS.results_dir, 'Fro-new-labels' + str(e)), Fro_new_labels)
"""
    
