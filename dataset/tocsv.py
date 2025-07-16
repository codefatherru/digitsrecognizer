# from https://github.com/AhmetTumis/mnist-png-to-csv-converter/blob/main/main.py

from PIL import Image, ImageOps
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

def png_to_csv(png):
    image_file = Image.open(png)
    inverted_img = ImageOps.invert(image_file)
    img_grey = inverted_img.convert('L') # convert image to grayscale
    #image_file.save('testimage.gif')

    value = np.asarray(img_grey.getdata(), dtype=np.integer ).reshape((img_grey.size[1], img_grey.size[0]))
    #print(value)
    # перевели матрицу в массив
    fvalue = value.flatten()
    #print(fvalue)
    return fvalue



files = {}

for dirname, dirnames, filenames in os.walk('.'):


    for filename in filenames:
        #print(dirname)
        #print(filename)
        dirs = dirname.split(os.path.sep)
        #print(dirs)

        if len(dirs) > 2 and '.png' in filename:
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
    #print((v))
    with open('mnist_'+ k +'.csv', "a") as f:
        np.savetxt(f, v, fmt="%d", delimiter=",")



