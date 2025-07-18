# from https://github.com/AhmetTumis/mnist-png-to-csv-converter/blob/main/main.py

import cv2
import numpy as np
import os



def roman_to_arabian(letter):
    if letter == 'x':
        return 10
    elif letter == 'ix':
        return 9
    elif letter == 'iv':
        return 4
    elif letter == 'v':
        return 5
    elif letter == 'vi':
        return 6
    elif letter == 'vii':
        return 7
    elif letter == 'viii':
        return 8
    elif letter == 'i':
        return 1
    elif letter == 'ii':
        return 2
    elif letter == 'iii':
        return 3
    return None

def png_to_csv(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # массив массивов [[255 255 255 ... 255 255 255]
    #print(img)
    gray = 255 - img # инверсия в негатив [[0 0 0 ... 0 0 0]
    #print(gray)
    gray = cv2.resize(gray, (28, 28)) # [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   00   0   0   0   0   0   0   0   0   0]
    #@todo понять разницу
    #print(gray)
    #return gray  # массив 28на28
    # перевели матрицу в массив
    fvalue = gray.flatten()
    #print('invert ', fvalue)
    #exit(8)
    return fvalue



files = {}

for dirname, dirnames, filenames in os.walk('.'):


    for filename in filenames:
        print(dirname)
        print(filename)
        dirs = dirname.split(os.path.sep)
        #print(dirs)

        if (dirname.startswith('.'+os.path.sep+'test') or dirname.startswith('.'+os.path.sep+'train') or dirname.startswith('.'+os.path.sep+'val') ) and len(dirs) > 2 and '.png' in filename:
            letter = dirs[-1]
            vol = dirs[-2]
            print(filename)
            #print(vol, letter)

            line = png_to_csv(os.path.join(dirname, filename))

            # вставим в первую позицию значение картинки
            arrayToDump = np.insert(line, 0, roman_to_arabian(letter))
            print(arrayToDump)
            if not (vol in files):
                #print(vol)
                files[vol] = []
            #print(len(files[vol]))
            files[vol].append(arrayToDump)

print('Запись в CSV')
for k, v in files.items():
    print(k)
    print(len(v))
    print((v))
    with open('mnist_'+ k +'.csv', "a") as f:
        np.savetxt(f, v, fmt="%d", delimiter=",")



