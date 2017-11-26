"""
Cifar-10 testing parameters
"""
import os

import keras
import numpy as np
from keras.datasets import cifar10
from keras.layers import Activation, Dense, regularizers, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, adam
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

def test_nn_initw(num_epochs=25, eta=1e-2, batch_size=64, hidden_neurons=100, ran_w=0.4, reg_str=0.4):
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
    in_layer = Dense(hidden_neurons, input_dim=3072,activation='relu',kernel_initializer=keras.initializers.RandomUniform(-ran_w, ran_w),
                     kernel_regularizer=regularizers.l2(reg_str))
    model.add(in_layer)
    model.add(Dropout(0.4))
    # 2nd hidden layer of 50 sigmoid neurons
    model.add(Dense(hidden_neurons,activation='relu',kernel_initializer=keras.initializers.RandomUniform(-ran_w, ran_w),
                    kernel_regularizer=regularizers.l2(reg_str)))
    model.add(Dropout(0.4))
    # output layer with 10 softmax neurons
    model.add(Dense(10, activation='softmax',kernel_initializer=keras.initializers.RandomUniform(-ran_w, ran_w),
                    kernel_regularizer=regularizers.l2(reg_str)))

    # optimizer object, with eta = 0.01
    #sgd = SGD(lr=eta)
    sgd = SGD(lr=eta, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["categorical_accuracy"])

    t = time.time()
    # show the progress during iteration
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_data=(x_valid, y_valid))
    elapsed = time.time() - t
    print("Training time: ", elapsed, "S")
    print ("Train acc: ", history.history['categorical_accuracy'][-1])
    print ("Validation acc: ", history.history['val_categorical_accuracy'][-1])
    return  model.evaluate(x_test, y_test,batch_size=batch_size, verbose=0)


#_err, valid_acc = test_nn_initw(num_epochs = 22, ran_w=0.003, hidden_neurons=50, eta=1e-2, batch_size=64, reg_str=0.01)
#print ("Error: ", _err, "\nTest Acc: ", valid_acc*100, "\n#####\n")

# _err, test_acc = test_nn_initw(num_epochs = 22, ran_w=0.003, hidden_neurons=50, eta=1e-3, batch_size=64, reg_str=0.01) # 40%    #decay=1e-6, momentum=0.9
# _err, test_acc = test_nn_initw(num_epochs = 32, ran_w=0.002, hidden_neurons=120, eta=1.5e-3, batch_size=80, reg_str=0.02) # 42% 

# _err, test_acc = test_nn_initw(num_epochs = 32, ran_w=0.002, hidden_neurons=40, eta=2.5e-3, batch_size=128, reg_str=0*0.02) # 42.4%  # decay=1e-4, momentum=0.9
# _err, test_acc = test_nn_initw(num_epochs = 32, ran_w=0.002, hidden_neurons=80, eta=2.5e-3, batch_size=128, reg_str=0*0.02) # 44.4% 
# _err, test_acc = test_nn_initw(num_epochs = 32, ran_w=0.002, hidden_neurons=80, eta=2.5e-3, batch_size=256, reg_str=0*0.02) # 45.95% 
# _err, test_acc = test_nn_initw(num_epochs = 32, ran_w=0.002, hidden_neurons=120, eta=2.5e-3, batch_size=256, reg_str=0*0.02) # 47.6% 
# _err, test_acc = test_nn_initw(num_epochs = 32, ran_w=0.002, hidden_neurons=150, eta=2.5e-3, batch_size=256, reg_str=0*0.02) # 48.4% 
# _err, test_acc = test_nn_initw(num_epochs = 80, ran_w=0.002, hidden_neurons=150, eta=2.5e-3, batch_size=512, reg_str=0*0.02) # 50.8% 
# print ("Error: ", _err, "\nTest Acc: ", test_acc*100, "\n#####\n")

_err, test_acc = test_nn_initw(num_epochs = 100, ran_w=0.002, hidden_neurons=150, eta=2.5e-3, batch_size=512, reg_str=0*0.02) # 50.8% 
print ("Error: ", _err, "\nTest Acc: ", test_acc*100, "\n#####\n")