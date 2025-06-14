# digits recognizer



## Getting Started

https://blog.bayrell.org/ru/iskusstvennyj-intellekt/418-raspoznavanie-czifry-po-baze-mnist.html

> python my.php

использует готовый датасет mnist в формате CSV

> wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz -O "mnist.npz"

To load this dataset in your code use following function

> def load_data(path):
>   with np.load(path) as f:
>       x_train, y_train = f['x_train'], f['y_train']
>       x_test, y_test = f['x_test'], f['y_test']
>       return (x_train, y_train), (x_test, y_test)
> 
> (x_train, y_train), (x_test, y_test) = load_data('../input/mnist.npz')

https://www.kaggle.com/datasets/vikramtiwari/mnist-numpy

##  Roman numerals 

??? https://agneevmukherjee.github.io/agneev-blog/preparing-a-Roman-MNIST/

https://www.kaggle.com/datasets/shubhamcodez/roman-number110-dataset?resource=download

https://www.kaggle.com/datasets/agneev/emnistbased-handwritten-roman-numerals?select=500_each_EMNIST-based-Roman

https://www.kaggle.com/datasets/agneev/yaromnist-dataset

https://www.kaggle.com/datasets/agneev/combined-handwritten-roman-numerals-dataset 

https://www.kaggle.com/datasets/agneev/basedonenglishhandwrittencharactersmodified 
