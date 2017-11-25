"""
Preprocessing script that will load tha cifar-10 data set
"""

import numpy as np

from data_utils import load_CIFAR10


def init_data(num_samples=5000, num_test=500, cifar10_dir='cifar-10-batches-py'):
    """
    Will load the entire cifar dataset and return the first <num_samples> items
    args:
     num_samples : the number of training sample items taken from the whole dataset
     num_test : the number of test items [the un-seen test items] taken from the dataset
                that will be used to test our classifier
     cifar10_dir : the directory contains the cifar dataset
    returns:
     array of <num_samples> items each of (32x32x3) ints representing pixles
     that is reshaped into a 1D array
    """
    # load the whole data-set
    x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)

    # trim the training and test data to fit in memory
    x_train = x_train[:num_samples] # array of pixles 32x32x3
    y_train = y_train[:num_samples] # array of lables
    x_test = x_test[:num_test]      # array of pixles 32x32x3
    y_test = y_test[:num_test]      # array of lables

    # reshape data
    # let the <num_samples> image objects stored in a 1D array i.e. <num_samples> element or arrays each of 32x32x3
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    # let the <num_test> img objects stored in a 1D array i.e. <num_test> element/arr each of 32x32x3=3072 item
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    # classes as seen in the cifar data set
    classes = np.array(['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog'\
    , 'horse', 'ship', 'truck'])
    #return reshapped data
    return x_train/255.0, y_train, x_test/255.0, y_test, classes

# def correct_classes(net_out, target_out):
#     """
#     Returns the number of the correct classified items
#     for the cifar-10 dataset. It supports batch calculation

#     args:
#     + net_out : the neural network output when the test/valid data is probed
#     + target_out : the neural network desired output for the current input(s)

#     returns:
#     + ret : the amount of correct classified classes
#     """
#     ret = np.sum(np.argmax(net_out, axis=1) == np.argmax(target_out, axis=1))
#     return ret
