"""
Defines all the activation functions and their derivatives

Dependencies
- Numpy

Currently supports
 - logistic sigmoid
"""

import numpy as np

def sigmoid(vect):
    """
    A vector sigmoid function 1/(1-exp(-vect)) that
    calculates sigmoid(vect)
    args:
        vect : a vector of floats
    returns:
        ret : array of 1/(1+exp(-vect))
    """
    return 1/(1+np.exp(-vect))

def sigmoid_d(vect):
    """
    A vector derivative sigmoid function sigmoid(vect)*(1-sigmoid(vect))
    args:
        vect : a vector of floats
    returns:
        ret : vector of d/dx sigmoid(vect)
    """
    s = sigmoid(vect)
    return s*(1-s)

def soft_max(vect):
    """
    A vector softmax function
    args :
        vect : a vector of floats
    returns:
        ret : array of exp(v[i])/sum(exp(v[i]))
    """
#    return np.exp(vect)/np.sum(np.exp(vect))
    ## safe exp:
    vect = np.exp(vect - np.max(vect))
    ret = vect/vect.sum() #np.exp(vect)/np.sum(np.exp(vect))
    return ret
