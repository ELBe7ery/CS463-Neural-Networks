"""
Cifar-10 neural network based classifier

Dependencies
    - FFNN.py
"""

from FFNN import FFNN
from cifar_init import init_data
import numpy as np
import pylab
################################
### Training variables
TRAIN_SIZE = 1000
TEST_SIZE = 100#2000
# Train for 10K epochs
EPOCHS = 30
# each iteration use a batch of 500 items
BATCH_SIZE = 1 #10#100#50
# learning rate 1e-1
LR = 1e-1

# create NN object with 2 hidden layers
# batch size : 500, learning rate : 1e-1
NN = FFNN([3072, 50, 50, 10], batch=BATCH_SIZE, eta=LR, cost_funct='mse')
################################

DATASET_DIR = 'F:\\Handasa\\Computer\\4th 7asbat\\Neural Networks\\labs\\KNN\\cifar-10-batches-py'
X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, CLASSES = \
        init_data(num_samples=TRAIN_SIZE, num_test=TEST_SIZE, cifar10_dir=DATASET_DIR)

# converting output into one hot encoding
Y_TRAIN = np.eye(10, 10)[Y_TRAIN] 
Y_TEST = np.eye(10, 10)[Y_TEST]

Y_AXIS = np.zeros([EPOCHS])

for i in range(EPOCHS):
    for j in range(0,TRAIN_SIZE,BATCH_SIZE):
        mini_batch_in = X_TRAIN[j:j+BATCH_SIZE, :].T
        mini_batch_target = Y_TRAIN[j:j+BATCH_SIZE, :].T
        NN.train_step(mini_batch_in, mini_batch_target)
    print "Epoch :", i
    acc = 0.0
    for jj in range(0,TEST_SIZE,BATCH_SIZE):
        #X_VALID  X_TRAIN
        acc += NN.test_acc(X_TEST[jj:jj+BATCH_SIZE,:].T, Y_TEST[jj:jj+BATCH_SIZE,:].T)
        #idx = np.random.randint(0,TEST_SIZE-BATCH_SIZE)
        #acc += NN.test_acc(X_TEST[idx:idx+BATCH_SIZE,:].T, Y_TEST[idx:idx+BATCH_SIZE,:].T)

    acc = 100*acc/TEST_SIZE
    print "Accuracy: ", acc, "%"
    Y_AXIS[i] = acc


pylab.plot(Y_AXIS)
pylab.show()

