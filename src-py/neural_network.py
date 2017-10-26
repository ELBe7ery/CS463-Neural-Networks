"""
Neural network class

Dependencies
-  Numpy
-  activation_functions
"""
import numpy as np
import activation_functions as actf


class Layer(object):
    """
    Implements a FULLY CONNECTED layer object, the neural network is defined
    as a chain of layers.
    ### args
    -    num_neurons : an int scaler value representing number of neurons,
            num row of the weight matrix
    -    num_inputs : an int scaler value representing the number of the inputs [or] neurons,
         num col of the weight matrix at the previous layer
    -    f_act : the activation function of all the neurons at the layer
    -    f_act_d : the derivative of the activation funct.
    -    out_neuron : flag specifing wheather this is an output neuron
         [useful for calculating delta]
    -    eta : learning rate

    ### attributes
    -    f_act
    -    f_act_d
    -    weight_matrix
    -    bias_vector
    -    input_vector
    -    out_vector
    -    delta_vector
    """
    def __init__(self, num_neurons, num_inputs, f_act=actf.sigmoid,
                 f_act_d=actf.sigmoid_d, out_neuron=False, eta = 0.001):
        """
        Creates a neural network layer
        """
        self.f_act = f_act
        self.f_act_d = f_act_d
        self.eta = eta

        #create a randomly intiallized weight, bias and net matrices
        self.weight_matrix = np.random.ranf([num_neurons, num_inputs])
        self.bias_vector = np.random.ranf([num_neurons, 1])
        self.net = np.zeros([num_neurons, 1])

        # create delta matrix, one delta per neuron
        self.delta_vector = np.zeros([num_neurons, 1])

        # create the input, output vectors
        self.input_vector = np.zeros([num_inputs, 1])
        self.out_vector = np.zeros([num_neurons, 1])

    def fwd_pass(self, in_vect):
        """
        Forward pass the input
        ### args
        -    in_vect : a float vector for the inputs to probe on the layer an Nx1 vector
        """
        self.input_vector = in_vect
        # mac output
        self.net = np.dot(self.weight_matrix, self.input_vector) + self.bias_vector
        # apply f_act
        self.out_vector = self.f_act(self.net)

    def calc_delta_out(self, target_vect):
        """
        Internal function for calculating the delta terms of the output layer
        ### args
        - target_vect : desired output, taken from the data-set label
        """
        self.delta_vector = (target_vect - self.out_vector)*self.f_act_d(self.net)#*self.input_vector.T

    def calc_delta_hidden(self, succ_layer):
        """
        Internal function for calculating the delta terms of the hidden layers
        ### args
        - past_layer : the succ. layer object
        """
        self.delta_vector = np.dot(succ_layer.weight_matrix.T, succ_layer.delta_vector)*self.f_act_d(self.net)


    def update_weights(self):
        """
        updates all the weights of the neural network
        """
        # broadcast
        self.weight_matrix += self.delta_vector*self.input_vector.T