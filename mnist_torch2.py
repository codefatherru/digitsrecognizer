# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
# Source: https://github.com/bayrell-tutorials/mnist
# Url:
# https://blog.bayrell.org/ru/iskusstvennyj-intellekt/411-obuchenie-mnogoslojnogo-perseptrona-operaczii-xor.html
##

import torch, os, math
import numpy as np

from torch import nn
from torch.utils.data import TensorDataset

from ai_helper import *


def data_normalize_x(data_x):
	r"""
	Нормализация датасета по x
	"""
	data_x = torch.from_numpy(data_x)
	data_x_shape_len = len(data_x.shape)
	
	if data_x_shape_len == 3:
		data_x = data_x.reshape(data_x.shape[0], -1)
	elif data_x_shape_len == 2:
		data_x = data_x.reshape(-1)
		
	data_x = data_x.to(torch.float32) / 255.0
	return data_x
	

def data_normalize_y(data_y):
	r"""
	Нормализация датасета по y
	"""
	data_y = list(map(get_vector_from_answer(10), data_y))
	data_y = torch.tensor( data_y )
	return data_y
	

class Network(AbstractNetwork):
	
	
	def __init__(self):
		
		AbstractNetwork.__init__(self)
		
		self.mnist_data = None
		
	
	def get_name(self):
		r"""
		Название сети
		"""
		return os.path.join("data", "model", "1")
	
	
	def load_dataset(self, type):
		
		r"""
		Загрузить датасет
		"""
		
		if self.mnist_data is None:
			self.mnist_data = np.load("data/mnist.npz", allow_pickle=True)
		
		# Обучающий датасет
		if type == "train":
			tensor = {}
			tensor["x_train"] = data_normalize_x(self.mnist_data["x_train"])
			tensor["y_train"] = data_normalize_y(self.mnist_data["y_train"])
			tensor["x_test"] = data_normalize_x(self.mnist_data["x_test"])
			tensor["y_test"] = data_normalize_y(self.mnist_data["y_test"])
			
			# Установить датасет
			self.train_dataset = TensorDataset( tensor["x_train"], tensor["y_train"] )
			self.test_dataset = TensorDataset( tensor["x_test"], tensor["y_test"] )
		
		# Контрольный датасет
		if type == "control":
			
			photo_number = 256
			
			x_train = np.asarray([ self.mnist_data["x_train"][photo_number] ])
			y_train = np.asarray([ self.mnist_data["y_train"][photo_number] ])
			x_train = data_normalize_x(x_train)
			y_train = data_normalize_y(y_train)
			
			self.control_dataset = TensorDataset( x_train, y_train )
	
	
	def check_answer(self, tensor_x=None, tensor_y=None, tensor_predict=None, type=None):
		"""
		Check answer
		"""
		
		y = get_answer_from_vector(tensor_y)
		predict = get_answer_from_vector(tensor_predict)
		
		if type == "control":
			print ("Model answer", predict)
			print ("Correct answer", y)				
			#plot_show_image(tensor_x, cmap='gray')
			
		return predict == y
		
	
	def create_model(self):
		
		AbstractNetwork.create_model(self)
		
		self.batch_size = 64
		
		self.model = nn.Sequential(
			nn.Linear(784, 128),
			nn.ReLU(),
			nn.Linear(128, 10),
			#nn.Softmax()
		)	
	
	
	def train_epoch_callback(self, **kwargs):
		r"""
		Train epoch callback
		"""
		
		epoch_number = kwargs["epoch_number"]
		loss_test = kwargs["loss_test"]
		accuracy_train = kwargs["accuracy_train"]
		accuracy_test = kwargs["accuracy_test"]
		
		if epoch_number >= 20:
			self.stop_training()
		
		if accuracy_train > 0.95:
			self.stop_training()
		
		if accuracy_test > 0.95:
			self.stop_training()
		
		if loss_test < 0.005 and epoch_number >= 5:
			self.stop_training()
	
	

if __name__ == '__main__':
	
	net = Network()
	
	
	# Создать модель
	net.create_model()
	net.summary()
	
	# Загрузить сеть с диска
	net.load()
	
	
	# Обучить сеть, если не обучена
	if not net.is_trained():
		
		# Загрузка обучающего датасета
		net.load_dataset("train")
		
		# Обучить сеть
		net.train()
		net.train_show_history()
		
		# Сохранить сеть на диск
		net.save()
	
	
	# Загрузка контрольного датасета
	net.load_dataset("control")
	
	# Проверка модели
	net.control()
	
