"""
Just for testing
"""

import FFNN
import numpy as np




# nn = Layer.Layer(3, 2)
# nn.weight_matrix = np.ones([3,3])
# nn.weight_matrix[:,1:] = 0.77
# nn.weight_matrix[:,0] = 0.67
# output_layer = Layer.Layer(3, 3)
# output_layer.weight_matrix = 0.5*np.ones([3,4])


# in_vect = np.array([1, 2]).reshape(2, 1)

# # fwd pass
# nn.fwd_pass(in_vect)
# output_layer.fwd_pass(nn.out_vector)

# print "outputs before 1st iteration"
# print output_layer.out_vector

# #back propagate error
# output_layer.calc_delta_out(np.array([1, 1, 1]).reshape(3,1))
# nn.calc_delta_hidden(output_layer)

# output_layer.update_weights()
# nn.update_weights()

# print "input weights after training"
# print nn.weight_matrix

# print "weight: "
# print output_layer.weight_matrix
# print "output: "
# print output_layer.out_vector


# nn._calc_delta_hidden(output_layer)

# print "delta"
# print output_layer.delta_vector

# print (output_layer.delta_vector[0,0]*output_layer.weight_matrix[0,0] 
# + output_layer.delta_vector[1,0]*output_layer.weight_matrix[1,0] 
# + output_layer.delta_vector[2,0]*output_layer.weight_matrix[2,0])

# print nn.delta_vector

input_vector = np.array([1,2]).T

net = FFNN.FFNN([2, 3, 3])

# adjust by hand :(
net.layers[0].weight_matrix = np.ones([3,3])
net.layers[0].weight_matrix[:,1:] = 0.77
net.layers[0].weight_matrix[:,0] = 0.67

net.layers[1].weight_matrix = 0.5*np.ones([3,4])

net.probe_input(input_vector, debug = True)

net.err_bp(np.ones([3,1]), debug=True)
net.probe_input(input_vector)
net.err_bp(np.ones([3,1]))
net.probe_input(input_vector)
net.err_bp(np.ones([3,1]), debug=True)