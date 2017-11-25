"""
Code that simulates a FFNN with two hidden layer
to classify cifar-10 data set

Assumes the data-set location is given and set correctly
In my case it is located at 'user\.keras\datasets\cifar-10-batches-py'
"""

import os

import keras
import numpy as np
from keras.datasets import cifar10
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.optimizers import SGD
import pylab

#### Loading data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

### One hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 50% validation, 50% test
x_valid = x_test[:5000, :]
y_valid = y_test[:5000, :]

x_test = x_test[5000:, :]
y_test = y_test[5000:, :]
####



### create a stack of layers
def test_nn(num_epochs=25, eta=1e-1, batch_size=128, hidden_neurons=50):
    """
    Creates a neural network FF and attempts to plot the training/valid loss and acc
    Note: i am using the test data as the validation data

    args:
    + num_epochs : the number of epochs to train
    + eta : learning rate
    + batch_size : the batch size
    """

    model = Sequential()

    # create an input layer followed by 50 sigmoid neurons
    in_layer = Dense(hidden_neurons, input_dim=3072,activation='sigmoid')
    model.add(in_layer)

    # 2nd hidden layer of 50 sigmoid neurons
    model.add(Dense(hidden_neurons,activation='sigmoid'))
    # output layer with 10 softmax neurons
    model.add(Dense(10, activation='softmax'))

    # optimizer object, with eta = 0.01
    sgd = SGD(lr=0.01)

    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["categorical_accuracy"])

    # show the progress during iteration
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0, validation_data=(x_valid, y_valid))
    pylab.plot(history.history['val_categorical_accuracy'])
    pylab.plot(history.history['categorical_accuracy'])
    pylab.legend(['Validation', 'Train'], loc='upper right')
    pylab.title('Neural Network Accuracy')
    pylab.xlabel('Epoch')
    pylab.ylabel('Accuracy')
    pylab.show()
    pylab.plot(history.history['val_loss'])
    pylab.plot(history.history['loss'])
    pylab.legend(['Validation', 'Train'], loc='upper right')
    pylab.title('Neural Network Loss')
    pylab.xlabel('Epoch')
    pylab.ylabel('Error')
    pylab.show()

    return  model.evaluate(x_test, y_test,batch_size=batch_size, verbose=0)

hidden_unit_size = [2, 5, 10, 20, 50, 100]
# a dictionary of index => num_neurons/layer that saves (err, acc)
result_dict = {}
for i in hidden_unit_size:
    print ("Running with hidden unit size: ", str(i))
    result_dict[i] = test_nn(num_epochs = 10, hidden_neurons=i)