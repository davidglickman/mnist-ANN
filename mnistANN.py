import keras
import numpy as np
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import  Sequential
from keras.datasets import mnist
from keras.utils import np_utils
from matplotlib import pyplot
from keras.utils import plot_model


(x_train, y_train), (x_test, y_test) = mnist.load_data()

xtestReshaped = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
xtrainReshaped = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
n_classes = 10
ytrainCategorical = np_utils.to_categorical(y_train, n_classes)
ytestCategorical = np_utils.to_categorical(y_test, n_classes)

mnistModel = Sequential()
mnistModel.add(Dense(784))
mnistModel.add(Activation('relu'))
mnistModel.add(Dense(387))
mnistModel.add(Activation('relu'))
mnistModel.add(Dense(10))
mnistModel.add(Activation('sigmoid'))
mnistModel.compile(loss='mean_squared_error', optimizer='sgd')
history = mnistModel.fit(xtrainReshaped, ytrainCategorical,
          batch_size=128, epochs=20,
          verbose=2,
          validation_data=(xtestReshaped, ytestCategorical))

print('\n# Evaluate on test data')
results = mnistModel.evaluate(xtestReshaped, ytestCategorical, batch_size=128)
print('test loss, test acc:', results)