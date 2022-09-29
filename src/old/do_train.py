#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, io, sys, torch
from tiny_ai_helper import *


def do_train():
	
	"""
	Do train model
	"""
	
	from src.model.Mnist4 import Mnist4
	
	model = Mnist4()
	model.create_model()
	model.summary()
	
	# Загрузить модель с диска
	#model.load()
	#model.save()
	
	# Обучить сеть, если не обучена
	if not model.is_trained():
		
		# Загрузка обучающего датасета
		model.load_dataset(type="train")
		
		# Обучить сеть
		model.train()
		model.show_train_history()
	
	pass


def do_answer():
	
	"""
	Do answer model
	"""
	
	from src.model.Mnist4 import Mnist4
	
	model = Mnist4()
	model.create_model()
	model.summary()
	
	# Загрузить модель с диска
	model.load()
	
	data_x = torch.tensor([])
	
	input_shape = [28, 28]
	t = torch.zeros(input_shape).to(torch.float32)
	t = t[None,:]
	
	data_x = torch.cat( (data_x, t) )
	data_x = torch.cat( (data_x, t) )
	data_x = torch.cat( (data_x, t) )
	data_x = torch.cat( (data_x, t) )
	data_x = torch.cat( (data_x, t) )
	
	ans = model.predict(data_x)
	
	print( ans.shape )
	


def main():
	"""
	Main app
	"""
	
	do_train()
	#do_answer()
	
	pass


if __name__ == '__main__':
	main()
	pass
