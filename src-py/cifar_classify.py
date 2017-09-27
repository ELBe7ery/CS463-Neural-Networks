"""
Classifier based on the KNN
"""
import numpy as np
import matplotlib.pyplot as plt

# Adjust plotter options
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class KNN():
    """
    KNN classifier class
    """

    def __init__(self, num_k=1, num_samples=5000, num_test=500, dataset_dir='cifar-10-batches-py'):
        """
        KNN constructor that loads the data set
        args :
         num_k : number of nerest neighbours to check for
         num_samples : the number of training sample items taken from the whole dataset
         num_test : the number of test items [the un-seen test items] taken from the dataset
                    that will be used to test our classifier
         dataset_dir : the directory contains the dataset

        class attributes :
            self.num_k : number of neighbours to check, used by the self.classify() method
            self.x_train : array of 1D pixles of length num_samples, representing the raw data to train the model
            self.y_train : the lables of the training data
            x_test : array of 1D pixles of length num_samples, representing the un-seen data to test the model
            y_test : test data lables
        """
        import cifar_init
        self.num_k = num_k
        self.x_train, self.y_train, self.x_test, self.y_test = cifar_init.init_data(num_samples, num_test, dataset_dir)
        

        