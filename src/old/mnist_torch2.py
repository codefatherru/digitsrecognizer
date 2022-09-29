# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
# Source: https://github.com/bayrell-tutorials/mnist
# Url:
# https://blog.bayrell.org/ru/iskusstvennyj-intellekt/411-obuchenie-mnogoslojnogo-perseptrona-operaczii-xor.html
##

import torch, os, math, sys
import numpy as np

from torch import nn
from torch.utils.data import TensorDataset

from ai_helper import *
from model.Mnist import Mnist
from model.Mnist2 import Mnist2
from model.Mnist3 import Mnist3
from model.Mnist4 import Mnist4

	
def check_model(net:Mnist):	
	dataset = net.get_control_dataset(count=2)
	net.debug(True)
	net.predict(dataset.tensors[0])
	sys.exit()
	
	

if __name__ == '__main__':
	
	net = Mnist4()
	
	print (net.get_name())
	
	# Создать модель
	net.create_model()
	net.summary()
	
	#check_model(net)
	
	# Загрузить сеть с диска
	net.load()
	#net._is_trained = False
	
	# Обучить сеть, если не обучена
	if not net.is_trained():
		
		# Загрузка обучающего датасета
		net.load_dataset(type="train")
		
		# Обучить сеть
		net.train()
		net.show_train_history()
		
		# Сохранить сеть на диск
		net.save()
	
	
	# Загрузка контрольного датасета
	net.load_dataset(type="control", count=512)
	
	# Проверка модели
	net.control()
	
	
	# Create onnx
	net.save_onnx()
	
	
	
