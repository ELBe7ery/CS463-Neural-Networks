"""
Neural network layer class

Dependencies
-  Numpy
-  activation_functions.py
"""
import numpy as np
import activation_functions as actf


class Layer(object):
    """
    Implements a FULLY CONNECTED layer object, the neural network is defined
    as a chain of layers. Each layer supports inserting multiple inputs at the same time
    for batch training. In case the argument batch_size is left, the layer assumes only one set of inputs
    otherwise the layer assumes multiple inputs [for batch training]
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
        -    batch_size : the batch size, specifies the number of cols at the input vector

    ### attributes
        -    f_act
        -    f_act_d
        -    weight_matrix
        -    input_vector
        -    out_vector
        -    delta_vector
        -    eta
        -    batch_size
    """
    def __init__(self, num_neurons, num_inputs, f_act=actf.sigmoid,
                 f_act_d=actf.sigmoid_d, eta=0.16, batch_size=1):
        """
        Creates a neural network layer
        """
        self.f_act = f_act
        self.f_act_d = f_act_d
        self.eta = eta
        self.batch_size = batch_size

        #create a randomly intiallized weight and net matrices
        self.weight_matrix = np.random.ranf([num_neurons, num_inputs+1]) #+1 for bias input at col 0
        self.net = np.zeros([num_neurons, 1])

        # create delta matrix, one delta per neuron
        self.delta_vector = np.zeros([num_neurons, 1])

        # create the input, output vectors
        # consider multiple inputs for batch training by specifying `batch_size` number of cols
        self.input_vector = np.zeros([num_inputs+1, batch_size]) #+1 for bias input of one at row 0

        # since multiple inputs within a batch corresponds to multiple outputs
        # let the output vector num cols = batch_size
        self.out_vector = np.zeros([num_neurons, batch_size])

    def fwd_pass(self, in_vect):
        """
        Forward pass the input
        ### args
        -    in_vect : a float vector for the inputs to probe on the layer an Nx1 vector
        """
        # set all the zero rows to one -> for bias input embedded within weight matrix
        self.input_vector[0, :] = 1
        # set all the input(s) starting from row 1 for all the cols [for batch]
        self.input_vector[1:, :] = in_vect#.T

        # mac output
        # now handles dotting weights with multiple inputs
        self.net = np.dot(self.weight_matrix, self.input_vector)
        # apply f_act
        # now handles SIMD f_act on multiple Net vectors
        self.out_vector = self.f_act(self.net)

    def calc_delta_out(self, target_vect):
        """
        Calculate the delta terms of the output layer
        ### args
        - target_vect : desired output, taken from the data-set label
        """
        self.delta_vector = (target_vect - self.out_vector)*self.f_act_d(self.net)#*self.input_vector.T

    def calc_delta_hidden(self, succ_layer):
        """
        Calculate the delta terms of the hidden layers
        ### args
        - past_layer : the succ. layer object
        """
        self.delta_vector = np.dot(succ_layer.weight_matrix[:, 1:].T,
                                   succ_layer.delta_vector)*self.f_act_d(self.net)


    def update_weights(self):
        """
        updates all the weights of the neural network
        """
        # broadcast
        # Sum the inputs in case of batch training
        self.weight_matrix += 1.0/self.batch_size*self.eta*np.dot(self.delta_vector, self.input_vector.T)
        #np.sum(self.delta_vector*self.input_vector.T, axis=1, keepdims=True)
        #np.sum(self.delta_vector, axis=1, keepdims=True)*\
                                              #np.sum(self.input_vector, axis=1, keepdims=True).T
        #self.eta*self.delta_vector*self.input_vector.T
