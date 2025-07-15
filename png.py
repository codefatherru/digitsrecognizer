# пробный файл для перевода png в csv

png = 'dataset/train/x/10_cap_1.png'

from PIL import Image, ImageOps
import PIL
import numpy as np
import matplotlib.pyplot as plt

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

# вставим в первую позицию значение картинки
arrayToDump = np.insert(fvalue, 0, 10)

print(arrayToDump)

