"""
Neural network layer class

Dependencies
-  Layer.py
    -  Numpy
    -  activation_functions.py
"""

from Layer import Layer


class FFNN(object):
    """
    Implements a feed forward neural network with fully connected layers
    the training algorithm used is the error back-propagation with gradient descent

    ### args
    - layers_structure : a vector of ints that contain the number of neurons of each layer
    the number of ints represents the number of layers
    e.g. [input_layer, hidd_1, hidd2, ... hidd_n, out]

    ### attributes
    - layers : All the **hidden layers** and the **output layer**,
      the input layer is not included here instead the inputs are probed into the first hidden layer
    """

    def __init__(self, layers_structure, batch=1):
        """
        Creates the layers, connectes them
        """

        #creates a chain of layers
        self.layers = [Layer(layers_structure[i], layers_structure[i-1], batch_size=batch)
                       for i in range(1, len(layers_structure))]

    def probe_input(self, in_vect, debug=False):
        """
        Probes an input vector to the neural network
        and updates self.layers[i].output_vector
        """
        # probe the inputs at the input layer and update the 1st hidden layer output vector
        self.layers[0].fwd_pass(in_vect)

        # fwd pass all the inputs from pred. till the output layer
        for i in range(1, len(self.layers)):
            self.layers[i].fwd_pass(self.layers[i-1].out_vector)
        if debug:
            print "Output layer : \n", self.layers[-1].out_vector

    def err_bp(self, target_out_vect, debug=False):
        """
        Back propagates the error given the target values from the data-set
        
        ### args
        - target_out_vect : the target values for the output neurons
        """
        # calculate delta for all output neurons
        self.layers[-1].calc_delta_out(target_out_vect)

        for i in range(2, len(self.layers)+1):
            # back-propagate the error succ. layer by layer
            self.layers[-i].calc_delta_hidden(self.layers[-i+1])
        
        # All the delta terms are calculated, update weights
        for i in range(len(self.layers)):
            self.layers[i].update_weights()

        if debug:
            print "New weights @input layer: \n", self.layers[0].weight_matrix

    def train_step(self, in_vect, target_out_vect):
        """
        A single training step, accepts a batch of inputs or a single input

        ### args
            - in_vect : a col of inputs, a batch contains multiple cols
            - target_out_vect : desired output vector, a column of output. A batch contains multipl
            cols
        """
        self.probe_input(in_vect)
        self.err_bp(target_out_vect)
