#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, io, sys, torch
from tiny_ai_helper import *
from src.model import *


def do_train(model:CoreModel):
	
	"""
	Do train model
	"""
	
	model.summary()
	
	# Загрузить модель с диска
	#model.load()
	
	# Обучить сеть, если не обучена
	if not model.is_trained():
		
		# Загрузка обучающего датасета
		model.load_dataset(type="train")
		
		# Обучить сеть
		model.train()
		model.show_train_history()
	
	pass


def do_saveonnx(model:CoreModel):
	
	model.load()
	
	if model.is_loaded():
		model.save_onnx()
	

def do_answer(model:CoreModel):
	
	model.load_dataset(type="train")
	
	index = 250
	vector_x = torch.tensor([])
	
	x = model.test_dataset[index][0]
	y = model.test_dataset[index][1]
	vector_x = append_tensor(vector_x, x)
	right_answer = get_answer_from_vector(y)
	
	answer = model.predict(vector_x)
	model_answer = get_answer_from_vector(answer[0])
	
	print ("Right answer: " + str(right_answer))
	print ("Model answer: " + str(model_answer), answer[0])
	show_image_in_plot(x * 255.0, cmap='gray')


def main():
	"""
	Main app
	"""
	
	model = create_model("4new")
	#model.summary()
	
	#do_train(model)
	#do_saveonnx(model)
	#do_answer(model)
	
	pass


if __name__ == '__main__':
	main()
	pass
