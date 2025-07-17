# from https://github.com/AhmetTumis/mnist-png-to-csv-converter/blob/main/main.py

import cv2
import numpy as np
import os
import math

def png_to_28(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gray = img #оставляем белый фон

    # применяем пороговую обработку
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # удаляем нулевые строки и столбцы
    while np.sum(gray[0]) == 0:
        gray = gray[1:]
    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)
    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]
    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)
    rows, cols = gray.shape


    # изменяем размер, чтобы помещалось в box 20x20 пикселей
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        gray = cv2.resize(gray, (cols, rows))

    gray = 255 - img

    #расширяем картинку до 28x28 пикселей, добавляя черные ряды и столбцы по краям
    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.pad(gray, (rowsPadding, colsPadding), 'constant')

    rez = cv2.imwrite(img_path.replace(".", '.'+os.path.sep+'28', 1), gray)


    return rez





files = {}

for dirname, dirnames, filenames in os.walk('.'):


    for filename in filenames:
        #print('__________________________')
        #print(dirname)
        #print(dirnames)
        #print(filename)
        dirs = dirname.split(os.path.sep)
        #print(dirs)

        if (dirname.startswith('.'+os.path.sep+'test') or dirname.startswith('.'+os.path.sep+'train') or dirname.startswith('.'+os.path.sep+'val') )and len(dirs) > 2  and '.png' in filename:
            letter = dirs[-1]
            vol = dirs[-2]
            #print(filename)
            png = os.path.join(dirname, filename)
            directory = dirname.replace(".", '.'+os.path.sep+'28', 1)

            if not os.path.exists(directory):
                os.makedirs(directory)

            if png_to_28(png):
                print(png)
            else:
                print('error ', png)




