#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT

import torch
from model.Mnist3 import Mnist3

net = Mnist3()
net.create_model()
net.load()

model = net.model
onnx_model_path = net.get_path_onnx()

# Save ONNX Model
tensor_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

input_shape = [1, 28, 28]

data_input = torch.zeros(input_shape).to(torch.float32)

model = model.to(tensor_device)
data_input = data_input.to(tensor_device)

torch.onnx.export(
	model,
	data_input,
	onnx_model_path,
	opset_version = 13,
	input_names = ['input'],
	output_names = ['output'],
	verbose=False
)
