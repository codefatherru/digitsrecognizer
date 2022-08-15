# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
# Source: https://github.com/bayrell-tutorials/mnist
# Url:
# https://blog.bayrell.org/ru/iskusstvennyj-intellekt/411-obuchenie-mnogoslojnogo-perseptrona-operaczii-xor.html
## 


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras


# Загружаем MNIST DataSet
# Скачайте архив:
# wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

# data_set2 = mnist.load_data()


# Загрузка датасета
def data_set_load():
	data = np.load("mnist.npz", allow_pickle=True)
	data_set = {
		"train": {
			"question": data["x_train"],
			"answer": data["y_train"],
		},
		"control": {
			"question": data["x_test"],
			"answer": data["y_test"],
		}
	}
	
	return data_set

	
# Вывод на экран информации о датасете	
def data_set_print_info(data_set):
	print ("Train images", data_set["train"]["question"].shape)
	print ("Train answers", data_set["train"]["answer"].shape)
	print ("Control images", data_set["control"]["question"].shape)
	print ("Control answers", data_set["control"]["answer"].shape)

	

# Тест цифры в датасете
def data_set_test(data_set, photo_number):
	print ("Number:", data_set["answer"][photo_number])
	plt.imshow(data_set["question"][photo_number], cmap='gray')
	plt.show()
	

	
def data_set_normalize_question(question):
	question = question.reshape(question.shape[0], -1)
	question = question.astype('float32') / 255.0
	return question
	

	
# Преобразует число в списке в выходной вектор
# Например:
# 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
def get_output_vector_by_number(number):
	
	res = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	
	if (number >=0 and number < 10):
		res[number] = 1.0
		
	return res
	
	
	
def data_set_normalize_answer(answer):
	answer = np.asarray( list(map(get_output_vector_by_number, answer)) )
	return answer
	


# Инициализация датасета
def data_set_init():
	
	data_set = data_set_load()
	# data_set_print_info(data_set)
	
	# Проверяем цифру	
	# data_set_test(data_set["train"], 128)
	
	data_set["train"]["question"] = data_set_normalize_question(data_set["train"]["question"])
	data_set["train"]["answer"] = data_set_normalize_answer(data_set["train"]["answer"])
	data_set["control"]["question"] = data_set_normalize_question(data_set["control"]["question"])
	data_set["control"]["answer"] = data_set_normalize_answer(data_set["control"]["answer"])

	return data_set
	
	

# Создание нейронной сети
def create_model():
	
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense, Input
	
	input_vector_size = 784
	output_vector_size = 10
	
	model = Sequential(name="mnist_model")
	
	# Входной слой
	model.add(Input(input_vector_size, name='input'))

	# Скрытые слои
	model.add(Dense(128, name='hidden1', activation='relu'))
	
	# Выходной слой. Функция активации Sigmoid
	model.add(Dense(output_vector_size, name='output', activation='softmax'))

	# Среднеквадратическая функция ошибки
	model.compile(
		loss='mean_squared_error', 
		optimizer='adam',
		metrics=['accuracy'])
	
	return model
	
	
	
# Создание нейронной сети
def show_model_info(model, file_name="model.png"):
	
	# Вывод на экран информация о модели
	model.summary()
	
	# Схема модели
	#keras.utils.plot_model(
	#	model,
	#	to_file=file_name,
	#	show_shapes=True,
	#	show_layer_names=True)
	
	
	
# Обучение модели
def train_model(model, data_set):
	
	history = model.fit(
		# Входные данные
		data_set["train"]["question"],
		
		# Выходные данные
		data_set["train"]["answer"],
		
		# Размер партии для обучения
		batch_size=128,
		
		# Количество эпох обучения
		epochs=2,
		
		# Контрольные данные
		validation_data=(data_set["control"]["question"], data_set["control"]["answer"]),
		
		# Подробный вывод
		verbose=1) 
	
	return history
	
	
# Создаем, обучаем модель и сохраняем
def create_model_train_and_save():
	
	# Загружаем датасет
	data_set = data_set_init()
	data_set_print_info(data_set)

	# Создание модели
	model = create_model()
	show_model_info(model)

	# Обучение модели
	history = train_model(model, data_set)

	# Сохраняем модель на диск
	model.save('model_mnist')
	
	# Сохраняем картинку
	plt.plot( np.multiply(history.history['accuracy'], 100), label='Правильные ответы')
	plt.plot( np.multiply(history.history['val_accuracy'], 100), label='Контрольные ответы')
	plt.plot( np.multiply(history.history['loss'], 100), label='Ошибка')
	plt.ylabel('Процент')
	plt.xlabel('Эпоха')
	plt.legend()
	plt.savefig('model_history.png')
	plt.show()
	
	return model
	
	
def load_model_from_save():
	model = keras.models.load_model('model_mnist')
	return model
	

# Распознаем картинку по изображению
def photo_recognize(model, photo):
	
	import math
	
	# Нормализация данных фотографии
	photo = photo.reshape(-1)
	photo = photo / 255.0
	
	# Спрашиваем у нейронной сети ответ
	test = np.expand_dims(photo, axis=0)
	answer = model.predict( test )
	
	# Узнаем ответ. Позиция максимального значения в векторе answer[0] будет ответом
	ans_max = -math.inf
	ans_index = 0
	for i in range(0, len(answer[0])):
		ans = answer[0][i]
		if ans_max < ans:
			ans_index = i
			ans_max = ans
			
	return ans_index
		
	
	
# Тестирование картинки
def photo_test(model, data_set, photo_number):
	
	photo = data_set["question"][photo_number]
	photo_correct_number = data_set["answer"][photo_number]
	
	model_answer = photo_recognize(model, photo)
	
	print ("Model answer", model_answer)
	print ("Correct answer", photo_correct_number)
	
	plt.imshow(photo, cmap='gray')
	plt.show()
	
	
#model = create_model_train_and_save()

data_set = data_set_load()
model = load_model_from_save()

photo_test(model, data_set["control"], 120)

