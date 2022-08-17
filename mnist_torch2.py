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


def save_onnx(net:Mnist):
	
	# Create onnx
	import onnx, torch, torch.onnx
	#from onnx_tf.backend import prepare
	
	onnx_model_path = "web/mnist2.onxx"
	#onnx_model_path = net.get_name() + ".onxx"
	#tf_model_path = net.get_name()
	
	data_input = torch.zeros(net.input_shape).to(torch.float32)
	data_input = data_input[None,:]
	torch.onnx.export(
		net.model,
		data_input,
		onnx_model_path,
		opset_version = 9,
		input_names = ['input'],
		output_names = ['output'],
		verbose=False
	)
	
	#onnx_model = onnx.load(onnx_model_path)
	#tf_model = prepare(onnx_model)
	#tf_model.export_graph(tf_model_path)
	
	
def check_model(net:Mnist):	
	dataset = net.get_control_dataset()
	net.debug(True)
	net.predict(dataset.tensors[0])
	
	
	

if __name__ == '__main__':
	
	net = Mnist2()
		
	# Создать модель
	net.create_model()
	net.summary()
	
	#check_model(net)
	
	#sys.exit()
	
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
	#net.load_dataset(type="control", count=512)
	
	# Проверка модели
	#net.control()
	
	
	# Create onnx
	save_onnx(net)
	
	
	
