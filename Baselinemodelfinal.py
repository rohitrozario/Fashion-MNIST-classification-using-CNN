import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model


ROWS = 28
COLS = 28
CLASSES = 10




def loadthedataset():
    # reshape dataset to have a single channel
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def prepdataset(train, test):
	# convert from integers to floats
	train_normal = train.astype('float32')
	test_normal = test.astype('float32')
	# normalize to range 0-1
	train_normal = train_normal / 255.0
	test_normal = test_normal / 255.0
	# return normalized images
	return train_normal, test_normal

#trainX, testX = prep_pixels(trainX, testX)

#X_train, X_val, y_train, y_val = train_test_split(trainX, trainY, test_size=TEST_SIZE, random_state=RANDOM_STATE)


#print("Fashion MNIST train -  rows:",X_train.shape[0]," columns:", X_train.shape[1:4])
#print("Fashion MNIST valid -  rows:",X_val.shape[0]," columns:", X_val.shape[1:4])
#print("Fashion MNIST test -  rows:",testX.shape[0]," columns:", testX.shape[1:4])


def create_CNNmodel():
    # Model
    model = Sequential()
    # Add convolution 2D
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='he_normal',
                     input_shape=(ROWS, COLS, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(CLASSES, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    return model



def runthemodel():
    trainX, trainY, testX, testY = loadthedataset()
    trainX, testX = prep_pixels(trainX, testX)
    model = create_CNNmodel()
    model.fit(trainX, trainY, epochs=10, batch_size=128, verbose=0)
    _, acc = model.evaluate(testX, testY, verbose=0)
    print((acc * 100.0))



runthemodel()
