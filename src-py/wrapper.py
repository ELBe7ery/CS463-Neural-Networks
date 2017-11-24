"""
Cifar-10 neural network based classifier

Dependencies
    - FFNN.py
"""

from FFNN import FFNN
from cifar_init import init_data

DATASET_DIR = 'F:\\Handasa\\Computer\\4th 7asbat\\Neural Networks\\labs\\KNN\\cifar-10-batches-py'

X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, CLASSES = \
        init_data(num_samples=25000, num_test=1500, cifar10_dir=DATASET_DIR)

#NN = FFNN([3072, 500, 500, 1])


