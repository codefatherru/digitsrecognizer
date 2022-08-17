#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT


import onnx, torch.onnx
from onnx_tf.backend import prepare
from ai_helper import *
from model.Mnist import Mnist
from model.Mnist2 import Mnist2

net = Mnist2()
net.create_model()
net.load()

#onnx_model_path = "web/mnist2.onxx"
onnx_model_path = net.get_name() + ".onxx"
tf_model_path = net.get_name() + ".pb"

data_input = torch.zeros(net.input_shape)
torch.onnx.export(
	net.model,
	data_input,
	onnx_model_path,
	opset_version=12
	input_names = ['input'],
	output_names = ['output'],
	verbose=False
)

onnx_model = onnx.load(onnx_model_path)
tf_model = prepare(onnx_model)
tf_model.export_graph(tf_model_path)
