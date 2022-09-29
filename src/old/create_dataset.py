# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, io, sys, torch, time
from src.helper import create_train_dataset
from tiny_ai_helper import *


def chunk_loader():
	
	image_size=(128,128)
	image_mode = "RGB"
	image_size_path = str(image_size[0]) + "x" + str(image_size[1]) + "x" + image_mode
	dataset_resize_path = os.path.join(os.getcwd(), "data", "train",
		"simpsons_dataset_" + image_size_path)
	
	train_chunk_loader = ChunkLoader()
	train_chunk_loader.set_prefix("train")
	train_chunk_loader.set_chunk_path(dataset_resize_path)
	train_chunk_loader.load_all_chunks()
	
	#print( "Chunk size: " + str(train_chunk_loader.chunk_size) )
	#print( "Total data: " + str(train_chunk_loader.total_data_count) )
	
	chunk = train_chunk_loader.load_chunk(0)
	print(chunk[0].shape)
	show_image_in_plot(chunk[0][0])


def main():
	"""
	Main app
	"""
	
	print ("Start create dataset after")
	for i in range(3, 0, -1):
		print (i)
		time.sleep(1)
	
	create_train_dataset(image_size=(128,128))
	
	#chunk_loader()
	
	pass


if __name__ == '__main__':
	main()
