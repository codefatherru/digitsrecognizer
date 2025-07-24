# digits recognizer

проект распознование цифр из MNIST при помощи torch

## Getting Started

https://blog.bayrell.org/ru/iskusstvennyj-intellekt/418-raspoznavanie-czifry-po-baze-mnist.html

> python my.php

использует готовый датасет mnist в формате CSV.
массивы ["x_train"],["y_train"],["x_test"],["y_test"]
по ключу Y - значение числа
по ключу X - массивы 28х28 чисел в оттенках серого 0-255

> wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz -O "mnist.npz"

To load this dataset in your code use following function

> def load_data(path):
>   with np.load(path) as f:
>       x_train, y_train = f['x_train'], f['y_train']
>       x_test, y_test = f['x_test'], f['y_test']
>       return (x_train, y_train), (x_test, y_test)
> 
> (x_train, y_train), (x_test, y_test) = load_data('../input/mnist.npz')

скрипт разбора PNG картинок в CSV. картинки лежат в dataset/*, результат сохраняется в dataset/mnist_test.csv и тд
> python .\tocsv.py 

### Ошибки
даёт 
>ValueError: setting an array element with a sequence. The requested array has an inhomogeneou
s shape after 1 dimensions. The detected shape was (985,) + inhomogeneous part.

при разборе 
8_cap_450.png
[8 0 0 ... 0 0 0]
8_cap_451.png
[8 0 0 ... 0 0 0]
8_cap_452.png
[8 0 0 ... 0 0 0]



https://www.kaggle.com/datasets/vikramtiwari/mnist-numpy

??? https://habr.com/ru/articles/668144/ Подгон под MNIST-овский датасет

###  приведение к MNIST

_Исходные MNIST-овские цифры помещаются в квадратную картинку 20x20 пикселей. Затем вычисляется центр масс изображения и оно располагается на поле размера 28x28 пикселей таким образом, чтобы центр масс совпадал с центром поля. Именно к такому виду мы и должны подгонять наши данные._

инвертируем, обрезаем пустые края, приводим к 28х28. результат лежит в папке ./dataset/28
> python img.py

https://habr.com/ru/articles/668144/

Попробуем взять за основу реализацию модельки для распознавания MNIST-овских чисел через tensorflow https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/ (копия сохранена)
важны комментарии к статье (Confirm you have Keras 2.3 and Tensorflow 2.0 installed.)

### ошибки
pip show tensorflow

pip install tensorflow==2.0
>tensorflow 2.0 and tensorflow 2.1 require Python 2.7 and 3.4-3.7 but not higher.

`Python39\lib\typing.py", line 215, in _remove_dups_flatten
    all_params = set(params)
TypeError: unhashable type: 'list'__
`

https://www.python.org/downloads/ 

курс https://github.com/spbu-math-cs/ml-course

Extract all bounding boxes - найти контур, ограничивающий цифру, взять его в качестве основного изображения и сделать resize до нужных размеров. Пример, как это можно делать. В том числе может пригодиться, если необходимо распознавать числа из более чем одной цифры
https://stackoverflow.com/questions/21104664/extract-all-bounding-boxes-using-opencv-python

скрипт перевода PNG картинок в MNIST формат. 28х28 негатив
> python .\to28.py 


##  Roman numerals 

??? https://agneevmukherjee.github.io/agneev-blog/preparing-a-Roman-MNIST/

??? https://github.com/AhmetTumis/mnist-png-to-csv-converter/blob/main/main.py

https://www.kaggle.com/datasets/shubhamcodez/roman-number110-dataset?resource=download

https://www.kaggle.com/datasets/agneev/emnistbased-handwritten-roman-numerals?select=500_each_EMNIST-based-Roman

https://www.kaggle.com/datasets/agneev/yaromnist-dataset

https://www.kaggle.com/datasets/agneev/combined-handwritten-roman-numerals-dataset 

https://www.kaggle.com/