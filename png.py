# пробный файл для перевода png в csv

png = "dataset\\train\\x\\10_cap_1.png"

from PIL import Image, ImageOps
import PIL
import numpy as np
import matplotlib.pyplot as plt
import os


def roman_to_arabian(letter):
    if letter == 'x':
        return 10
    elif letter == 'v':
        return 5
    return None

image_file = Image.open(png)
inverted_img = ImageOps.invert(image_file)
img_grey = inverted_img.convert('L') # convert image to grayscale
#image_file.save('testimage.gif')

value = np.asarray(img_grey.getdata(), dtype=np.integer ).reshape((img_grey.size[1], img_grey.size[0]))
print(value)
# перевели матрицу в массив
fvalue = value.flatten()
print(fvalue)

#отобразим матрицу
plt.imshow(value, cmap='gray')
plt.show()



pre = ''
if 'train' in png:
    pre = 'train'
elif 'test' in png:
    pre = 'test'

print(png)
print(os.path.sep)
label = png[(png.rfind(os.path.sep) - 1)]

dirs = png.split(os.path.sep)
print(dirs)

print(dirs[-2])

print(label)



# вставим в первую позицию значение картинки
arrayToDump = np.insert(fvalue, 0, roman_to_arabian(dirs[-2]))

print(arrayToDump)

