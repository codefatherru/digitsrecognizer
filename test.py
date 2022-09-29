#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-


import torch, time, math, os


def test1():
	data = torch.tensor([
		[
			[1, 2],
			[3, 4],
		],
		[
			[1, 2],
			[3, 4],
		]
	])


	#print( data.reshape(data.shape[0], -1) )

	#for i in range(1000):	
	#	print ("\r" + str(i), end='')	
	#	time.sleep( 0.1 )	

	print ( round(0.7) )



def test2():
	
	t1 = torch.tensor([1, 2, 3])
	
	t1 = t1[None, :]
	
	print (t1)
	
	
def test3():
	
	t = "/drive/d/files/Projects/AI/mnist/data/model/4.zip"
	dir_name = os.path.dirname(t)
	
	file_name, _ = os.path.splitext(t)
	file_name = file_name + ".png"
	
test3()