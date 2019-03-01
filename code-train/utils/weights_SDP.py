"""
File contains all the preprocessing for weight matricees
"""

import numpy as np

"""  
Takes as input a matrix that is asymmetric
and returns the symmetric version for computing \infty to 1
"""

def weights_SDP(input_matrix):
    ## TODO: Fill in the code here
    num_rows = np.shape(input_matrix)[0]
    num_cols = np.shape(input_matrix)[1]

    Final_matrix = np.zeros([num_rows + num_cols, num_rows + num_cols])
    Final_matrix[0:num_rows, num_rows:num_rows + num_cols] = input_matrix
    Final_matrix[num_rows : num_rows + num_cols, 0:num_rows] = np.transpose(input_matrix)
    
    return Final_matrix



"""
improved weights SDP has both the linear 
and quadratic terms in the SDP while optimizing 
"""

def improved_weights_SDP(input_matrix, input_vector):
    
    num_rows = np.shape(input_matrix)[0]
    num_cols = np.shape(input_matrix)[1]
    Final_matrix = np.zeros([num_rows + num_cols + 1, num_rows + num_cols + 1])
    Final_matrix[1:num_rows+1, num_rows + 1:num_rows + num_cols + 1] = input_matrix
    Final_matrix[num_rows +1 : num_rows + num_cols + 1, 1:num_rows +1 ] = np.transpose(input_matrix)
    Final_matrix[1 + num_rows: 1 + num_rows + num_cols, 0] = np.ravel(input_vector)
    Final_matrix[0, 1+num_rows:1 + num_rows + num_cols] = np.ravel(input_vector)
    return Final_matrix
