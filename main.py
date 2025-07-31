#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## сеть из статьи https://habr.com/ru/articles/668144/

import sys
if sys.version_info[0:2] != (3, 7):
    raise Exception('Requires python 3.7.8')

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD



import cv2
import numpy as np


def rec_digit(img_path):
	global model

	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	gray = 255 - img

	gray = cv2.resize(gray, (28, 28))
	cv2.imwrite('gray' + img_path, gray)
	img = gray / 255.0
	img = np.array(img).reshape(-1, 28, 28, 1)
	out = str(np.argmax(model.predict(img)))
	return out


# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	print(trainX) # массив массивов
	# [[[0 0 0 ... 0 0 0]
	# [0 0 0 ... 0 0 0]

	print(trainX.shape) #(60000, 28, 28)
	print(trainX.shape[0]) #60000
	print(trainY) #[5 0 4 ... 5 6 8]
	print(trainY.shape) #(60000,)
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY) #Преобразует вектор класса (целые числа) в двоичную классную матрицу
	print(trainY) #[[0. 0. 0. ... 0. 0. 0.]
	# [1. 0. 0. ... 0. 0. 0.]
	print(trainY.shape) #(60000, 10)
	#exit(3)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY


# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm


# define cnn model
def define_model():
	global model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()
	# fit model
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=1)
	# save model
	model.save('digit_model.h5')
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))

	r = rec_digit('.\\dataset\\train\\v\\5white.png')
	#r = rec_digit('.\\dataset\\train\\viii\\8.png')
	print(r)

model = None

# entry point, run the test harness
run_test_harness()

