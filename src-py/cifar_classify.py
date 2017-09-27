"""
Classifier based on the KNN
"""
import numpy as np
import matplotlib.pyplot as plt

# Adjust plotter options
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class KNN(object):
    """
    KNN classifier class

    class attributes :
            self.num_k : number of neighbours to check, used by the self.classify() method
            self.x_train : array of 1D pixles of length num_samples,
                           representing the raw data to train the model
            self.y_train : the lables of the training data
            x_test : array of 1D pixles of length num_samples,
                     representing the un-seen data to test the model
            y_test : test data lables
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
        """
        import cifar_init
        self.num_k = num_k
        self.x_train, self.y_train, self.x_test, self.y_test, self.classes \
            = cifar_init.init_data(num_samples, num_test, dataset_dir)

    def show_from_test(self, test_id, debug=False):
        """
        Plots the image of self.x_test[test_id] this is to visulize a specific test image before
        using KNN to classify it
        args :
         test_id : the index of the test item to view
        """
        if debug:
            print "showing test image number", test_id,\
            "Which is a ", self.classes[self.y_test[test_id]]
        plt.imshow(self.x_test[test_id].reshape((32, 32, 3)).astype('uint8'))
        plt.show()

    def classify(self, test_id):
        """
        Predict an image given its dataset index
        args:
         test_id : the test data set index to test
        returns :
         class_name : a string representing the predicted class
        """
        score_arr = np.array([0]*(self.classes.size))
        sub_arr = np.sqrt(np.sum((self.x_train-self.x_test[test_id]) ** 2, axis=1))
        # take the k low nearest results
        sub_arr = sub_arr.argsort()[:self.num_k]
        # increment the repreated classes by 1 in the score array
        np.add.at(score_arr, self.x_test[sub_arr],  1)

        # take the highest scrore
        return self.classes[sub_arr.argsort[-1]]
