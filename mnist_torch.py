# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
# Source: https://github.com/bayrell-tutorials/mnist
# Url:
# https://blog.bayrell.org/ru/iskusstvennyj-intellekt/411-obuchenie-mnogoslojnogo-perseptrona-operaczii-xor.html
## 

import torch, os, sys, math
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset


# Загружаем MNIST DataSet
# Скачайте архив:
# mkdir data
# wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz -O "data/mnist.npz"

# data2 = mnist.load_data()

tensor_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  


def data_load():
	r"""
	Загрузка датасета
	"""
	data = np.load("data/mnist.npz", allow_pickle=True)
	data = {
		"train": {
			"x": data["x_train"],
			"y": data["y_train"],
		},
		"test": {
			"x": data["x_test"],
			"y": data["y_test"],
		}
	}
	
	return data


def data_print_info(data):
	r"""
	Вывод на экран информации о датасете	
	"""
	print ("Train images", data["train"]["x"].shape)
	print ("Train answers", data["train"]["y"].shape)
	print ("Test images", data["test"]["x"].shape)
	print ("Test answers", data["test"]["y"].shape)
	
	
def data_image_show(data, photo_number):
	r"""
	Тест цифры в датасете
	"""
	print ("Number:", data["y"][photo_number])
	plt.imshow(data["x"][photo_number], cmap='gray')
	plt.show()
	
	
def get_answer_from_vector(vector):
	r"""
	Returns answer from vector
	"""
	value_max = -math.inf
	value_index = 0
	for i in range(0, len(vector)):
		value = vector[i]
		if value_max < value:
			value_index = i
			value_max = value
	
	return value_index
	
	
def get_vector_by_number(count):
	
	r"""
	Преобразует число в списке в выходной вектор
	Например:
	1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
	5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
	"""
	
	def f(number):
		res = [0.0] * count
		
		if (number >=0 and number < count):
			res[number] = 1.0
			
		return res
		
	return f
	
	
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
	data_y = list(map(get_vector_by_number(10), data_y))
	data_y = torch.tensor( data_y )
	return data_y
	
	
def data_create():
	
	"""
	Создание дата сета
	"""
	
	data = data_load()
	data_print_info(data)
	
	# Проверяем цифру
	#data_image_show(data["train"], 512)
	
	data["train"]["x"] = data_normalize_x(data["train"]["x"])
	data["train"]["y"] = data_normalize_y(data["train"]["y"])
	data["test"]["x"] = data_normalize_x(data["test"]["x"])
	data["test"]["y"] = data_normalize_y(data["test"]["y"])
	
	return data


def create_model():
	
	"""
	Создание модели
	"""
	
	input_shape = 784
	output_shape = 10

	model = nn.Sequential(
		nn.Linear(input_shape, 128),
		nn.ReLU(),
		nn.Linear(128, output_shape),
		#nn.Softmax()
	)
	
	# Adam optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
	
	# mean squared error
	loss = nn.MSELoss()
	
	# Show model info
	summary(model)
	
	return {
		"input_shape": input_shape,
		"output_shape": output_shape,
		"model": model,
		"optimizer": optimizer,
		"loss": loss,
	}
	
	
def train_model(model_info, data):
	
	"""
	Обучение модели
	"""
	
	epochs = 20
	batch_size = 128
	history = {
		"loss_train": [],
		"loss_test": [],
	}
	
	model = model_info["model"]
	optimizer = model_info["optimizer"]
	loss = model_info["loss"]
	
	model = model.to(tensor_device)
	
	train_dataset = TensorDataset( data["train"]["x"], data["train"]["y"] )
	test_dataset = TensorDataset( data["test"]["x"], data["test"]["y"] )
	
	train_count = data["train"]["x"].shape[0]
	test_count = data["test"]["x"].shape[0]
	
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size, drop_last=True
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size, drop_last=True
	)
	
	for step_index in range(epochs):
		
		loss_train = 0
		loss_test = 0
		
		batch_iter = 0
		
		# Обучение
		for batch_x, batch_y in train_loader:
			
			batch_x = batch_x.to(tensor_device)
			batch_y = batch_y.to(tensor_device)
			
			# Вычислим результат модели
			model_res = model(batch_x)
			
			# Найдем значение ошибки между ответом модели и правильными ответами
			loss_value = loss(model_res, batch_y)
			loss_train = loss_value.item()
			
			# Вычислим градиент
			optimizer.zero_grad()
			loss_value.backward()
			
			# Оптимизируем
			optimizer.step()
			
			# Очистим кэш CUDA
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			
			del batch_x, batch_y
			
			batch_iter = batch_iter + batch_size
			batch_iter_value = round(batch_iter / train_count * 100)
			print (f"\rStep {step_index+1}, {batch_iter_value}%", end='')
		
		
		# Вычислим ошибку на тестовом датасете
		for batch_x, batch_y in test_loader:
			
			batch_x = batch_x.to(tensor_device)
			batch_y = batch_y.to(tensor_device)
			
			# Вычислим результат модели
			model_res = model(batch_x)
			
			# Найдем значение ошибки между ответом модели и правильными ответами
			loss_value = loss(model_res, batch_y)
			loss_test = loss_value.item()
			
		
		# Отладочная информация
		#if i % 10 == 0:
		print ("\r", end='')
		print (f"Step {step_index+1}, loss: {loss_train},\tloss_test: {loss_test}")
		
		# Остановим обучение, если ошибка меньше чем 0.01
		if loss_test < 0.01:
			break
		
		# Добавим значение ошибки в историю, для дальнейшего отображения на графике
		history["loss_train"].append(loss_train)
		history["loss_test"].append(loss_test)
		
	return history
	
	
def show_history(history):
	
	r"""
	Показать график обучения
	"""
	
	import matplotlib.pyplot as plt
	
	plt.plot( np.multiply(history['loss_train'], 100), label='Ошибка обучения')
	plt.plot( np.multiply(history['loss_test'], 100), label='Ошибка на тестах')
	plt.ylabel('Процент')
	plt.xlabel('Эпоха')
	plt.legend()
	plt.savefig('data/model_torch_history.png')
	plt.show()
	


def photo_test(model_info, data, photo_number):
	
	"""
	Тестирование картинки
	"""
	
	model = model_info["model"]
	photo = data["x"][photo_number]
	correct_answer = data["y"][photo_number]
	
	tensor_x = data_normalize_x(photo)
	tensor_x = tensor_x[None, :]
	tensor_y = model(tensor_x)
	
	model_answer = get_answer_from_vector(tensor_y[0].tolist())
		
	print ("Model answer", model_answer)
	print ("Correct answer", correct_answer)
	
	plt.imshow(photo, cmap='gray')
	plt.show()

	
	
model_path = "data/model_torch.zip"
model_info = create_model()
model = model_info["model"]

if not os.path.isfile(model_path):
	
	data = data_create()
	
	history = train_model(model_info, data)
	show_history(history)
	torch.save(model.state_dict(), model_path)
	
	del data
	
else:
	
	model.load_state_dict(torch.load(model_path))
	model.eval()


data = data_load()
photo_test(model_info, data["test"], 512)
