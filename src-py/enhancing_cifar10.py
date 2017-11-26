"""
Cifar-10 testing parameters
"""
import os

import keras
import numpy as np
from keras.datasets import cifar10
from keras.layers import Activation, Dense, regularizers
from keras.models import Sequential
from keras.optimizers import SGD
import pylab
import time

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

def test_nn_initw(num_epochs=25, eta=1e-2, batch_size=64, hidden_neurons=100, ran_w=0.4):
    """
    Creates a neural network FF and attempts to plot the training/valid loss and acc
    with regularization

    args:
    + num_epochs : the number of epochs to train
    + eta : learning rate
    + batch_size : the batch size
    """

    model = Sequential()

    # create an input layer followed by 50 sigmoid neurons
    in_layer = Dense(hidden_neurons, input_dim=3072,activation='sigmoid',kernel_initializer=keras.initializers.RandomUniform(-ran_w, ran_w))
    model.add(in_layer)

    # 2nd hidden layer of 50 sigmoid neurons
    model.add(Dense(hidden_neurons,activation='sigmoid',kernel_initializer=keras.initializers.RandomUniform(-ran_w, ran_w)))
    # output layer with 10 softmax neurons
    model.add(Dense(10, activation='softmax',kernel_initializer=keras.initializers.RandomUniform(-ran_w, ran_w)))

    # optimizer object, with eta = 0.01
    sgd = SGD(lr=eta)

    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["categorical_accuracy"])

    t = time.time()
    # show the progress during iteration
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0, validation_data=(x_valid, y_valid))
    elapsed = time.time() - t
    print("Training time: ", elapsed, "S")
    print ("Train acc: ", history.history['categorical_accuracy'][-1])
    print ("Validation acc: ", history.history['val_categorical_accuracy'][-1])

    pylab.plot(history.history['val_categorical_accuracy'])
    pylab.plot(history.history['categorical_accuracy'])
    pylab.legend(['Validation', 'Train'], loc='upper left')
    pylab.title('Neural Network Accuracy')
    pylab.xlabel('Epoch')
    pylab.ylabel('Accuracy')
    pylab.show()
    pylab.plot(history.history['val_loss'])
    pylab.plot(history.history['loss'])
    pylab.legend(['Validation', 'Train'], loc='upper left')
    pylab.title('Neural Network Loss')
    pylab.xlabel('Epoch')
    pylab.ylabel('Error')
    pylab.show()
    return  model.evaluate(x_test, y_test,batch_size=batch_size, verbose=0)
