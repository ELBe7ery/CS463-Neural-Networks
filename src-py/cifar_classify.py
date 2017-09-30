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
            num_samples : the number of training items
            num_test : the number of test items
            dist_arr : the array of distances for all test images wrt to all the training images
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
        self.num_samples = num_samples
        self.num_test = num_test
        self.dist_arr = None
        self.x_train, self.y_train, self.x_test, self.y_test, self.classes \
            = cifar_init.init_data(num_samples, num_test, dataset_dir)

    def show_from_test(self, test_id, debug=False):
        """
        Plots the image of self.x_test[test_id] this is to visulize a specific test image before
        using KNN to classify it
        args :
         test_id : the index of the test item to view
        """
        plt.imshow(self.x_test[test_id].reshape((32, 32, 3)).astype('uint8'))
        plt.show()

    def classify(self, test_id, view=False, internal=False, debug=False):
        """
        Predict an image given its dataset index
        args:
         test_id : the test data set index to test
         view : plot the image shape
        returns :
         class_name : a string representing the predicted class
        """
        score_arr = np.array([0]*(self.classes.size))
        # calculate the distance matrix by (self.x_train-self.x_test[test_id]) ** 2
        sub_arr = np.sqrt(np.sum((self.x_train-self.x_test[test_id]) ** 2, axis=1))
        # take the k low nearest results
        sub_arr = sub_arr.argsort()[:self.num_k]
        # increment the repreated classes by 1 in the score array
        np.add.at(score_arr, self.y_train[sub_arr], 1)
        if view:
            self.show_from_test(test_id)
        if debug:
            print "showing test image number", test_id,\
            "Which is a ", self.classes[self.y_test[test_id]]
        # take the highest scrore
        if internal:
            return score_arr.argsort()[-1]
        return self.classes[score_arr.argsort()[-1]]
    
    def accuracy(self, view_console=False):
        """
        Returns the model accuracy in percentage
        """
        correct = 0.0
        l = self.x_test.shape[0]
        for i in range(l):
            if(self.classify(i,internal=True) == self.y_test[i]):
                correct += 1
        ret = correct/l * 100
        if view_console:
            print ret
        return ret

    def calc_dist_test(self):
        """
        Calculates the whole distance of all the test items with respect
        to all the training data set.

        If n is the length of the training items, m is the length of the test items
        we need to return an nxm array where elements at row i and col j represent the
        distance of test item j from training item i.
        The code follow the equation of sum((test-train)**2) = sum(test**2) + sum(train**2) - 2*test*train`

        All the matricies in the above equation should be reshaped to be all n*m
        so since
         sum(test**2) will return n*1 matrix, we will replicate it m times  => n*m
         sum(test**2) will return m*1 matrix, we will transpose it => 1*m
          and replicate it n times n*m
         -2*test*train` will multiply n*m by n*m
           resulting in an n*m matrix

        return:
         array of size len(train)*len(test) such that the elements
         of row x represent all the test distanes from the training sample x
        """
        # a matrix of 1*m representing the sum of pixles of each item @ the test set
        sigma_test_pwr2 = np.sum(self.x_test**2, axis=1)

        # repeate this matrix n times
        sigma_test_pwr2 = np.tile(sigma_test_pwr2, (self.num_samples, 1))  # now sum(test**2) is an n*m matrix
        
        # a matrix of n*1 representing the sum of pixles of each item @ the training set
        sigma_train_pwr2 = np.sum(self.x_train**2, axis=1).reshape((self.num_samples, 1))

        # repeate this matrix m times
        sigma_train_pwr2 = np.tile(sigma_train_pwr2, (1, self.num_test))    # now sum(train**2) is an n*m matrix

        n_two_a_dot_b = -2 * self.x_train.dot(self.x_test.T)

        self.dist_arr = np.sqrt(sigma_train_pwr2 + sigma_test_pwr2 + n_two_a_dot_b)
        None

    def calc_acc(self):
        """
        calculates the accuracy, hopefully faster
        works at self.dist_arr n*m matrix
        """
        self.calc_dist_test()

        #for i in range()
