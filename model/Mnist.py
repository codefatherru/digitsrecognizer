# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import torch, os, math, random
import numpy as np

from torch import nn
from torch.utils.data import TensorDataset

from ai_helper import *



def data_normalize_x(data_x):
	r"""
	Нормализация датасета по x
	"""
	data_x = torch.from_numpy(data_x)
	
	"""
	data_x_shape_len = len(data_x.shape)
	
	if data_x_shape_len == 3:
		data_x = data_x.reshape(data_x.shape[0], -1)
	elif data_x_shape_len == 2:
		data_x = data_x.reshape(-1)
	"""
	
	data_x = data_x.to(torch.float32) / 255.0
	return data_x
	

def data_normalize_y(data_y):
	r"""
	Нормализация датасета по y
	"""
	data_y = list(map(get_vector_from_answer(10), data_y))
	data_y = torch.tensor( data_y )
	return data_y
	
	

class Mnist(AbstractNetwork):
	
	
	def __init__(self):
		
		AbstractNetwork.__init__(self)
		
		self.mnist_data = None
		
	
	def get_name(self):
		r"""
		Название сети
		"""
		return os.path.join("data", "model", "1")
	
	
	def get_train_dataset(self, **kwargs):
		
		"""
		Returns normalized train and test datasets
		"""
		
		if self.mnist_data is None:
			self.mnist_data = np.load("data/mnist.npz", allow_pickle=True)
		
		tensor = {}
		tensor["x_train"] = data_normalize_x(self.mnist_data["x_train"])
		tensor["y_train"] = data_normalize_y(self.mnist_data["y_train"])
		tensor["x_test"] = data_normalize_x(self.mnist_data["x_test"])
		tensor["y_test"] = data_normalize_y(self.mnist_data["y_test"])
		
		train_dataset = TensorDataset( tensor["x_train"], tensor["y_train"] )
		test_dataset = TensorDataset( tensor["x_test"], tensor["y_test"] )
		
		return train_dataset, test_dataset
	
	
	
	def get_control_dataset(self, **kwargs):
		
		"""
		Returns normalized control dataset
		"""
		
		if self.mnist_data is None:
			self.mnist_data = np.load("data/mnist.npz", allow_pickle=True)
		
		data_x = torch.tensor([])
		data_y = torch.tensor([])
		
		count = 0
		if "count" in kwargs:
			count = kwargs["count"]
		
		from random import shuffle
		data_list = [ i for i in range(self.mnist_data["x_test"].shape[0]) ]
		shuffle(data_list)
		
		for i in range(count):
			
			photo_number = data_list[i]
			
			x = np.asarray([ self.mnist_data["x_test"][photo_number] ])
			y = np.asarray([ self.mnist_data["y_test"][photo_number] ])
			x = data_normalize_x(x)
			y = data_normalize_y(y)
			
			data_x = torch.cat( (data_x, x) )
			data_y = torch.cat( (data_y, y) )
			
		return TensorDataset( data_x, data_y )
	
	
	
	def check_answer(self, **kwargs):
		"""
		Check answer
		"""
		
		type = kwargs["type"]
		tensor_x = kwargs["tensor_x"]
		tensor_y = kwargs["tensor_y"]
		tensor_predict = kwargs["tensor_predict"]
		
		tensor_x = tensor_x.reshape((28,28)).tolist()
		tensor_y = tensor_y.tolist()
		tensor_predict = tensor_predict.round().tolist()
		
		y = get_answer_from_vector(tensor_y)
		predict = get_answer_from_vector(tensor_predict)
		
		if type == "control":
			if predict != y:
				print (str(predict) + "|" + str(y))
				#print ("Correct answer", y)
				#show_image_in_plot(tensor_x, cmap='gray')
			
		return predict == y
		
	
	def create_model(self):
		
		AbstractNetwork.create_model(self)
		
		self.max_acc = 0.95
		self.max_epochs = 10
		self.batch_size = 64
		self.input_shape = (784)
		self.output_shape = (10)
		
		self.model = nn.Sequential(
			nn.Linear(784, 128),
			nn.ReLU(),
			nn.Linear(128, 10),
			#nn.Softmax()
		)