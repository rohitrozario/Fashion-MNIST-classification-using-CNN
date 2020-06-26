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
    model.add(Conv2D(64,
                     kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(CLASSES, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    return model



def evaluate_kfold(dataX, dataY, n_folds=4):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model = create_CNNmodel()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=128, validation_data=(testX, testY), verbose=0)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> ' % (acc * 100.0))
        # append scores
        scores.append(acc)
		 histories.append(history)
    return scores, histories

def results(scores):
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))

def runthemodel():
    trainX, trainY, testX, testY = loadthedataset()
    trainX, testX = prepdataset(trainX, testX)
    scores, histories = evaluate_kfold(trainX, trainY)
    results(scores)

runthemodel()
