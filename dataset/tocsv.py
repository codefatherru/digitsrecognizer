# from https://github.com/AhmetTumis/mnist-png-to-csv-converter/blob/main/main.py

import numpy as np
from PIL import Image
import os
import csv
from pprint import pprint
import matplotlib.pyplot as plt

def file_indexer():
    files = []
    for dirname, dirnames, filenames in os.walk('.'):
        for subdirname in dirnames:
            files.append(os.path.join(dirname, subdirname))

        for filename in filenames:
            files.append(os.path.join(dirname, filename))
    print('file_indexer:')
    pprint(files)

    writeToFile = []
    writeToTestFile = []

    for file in files:
        if ".png" in file and "train" in file:
            writeToFile.append(file)
        if ".png" in file and "test" in file:
            writeToTestFile.append(file)

    with open("Files.csv", "w") as f:
        write = csv.writer(f)
        write.writerows(writeToFile)
        print('writeToFile')
        pprint(writeToFile)

    with open("TestFiles.csv", "w") as f:
        write = csv.writer(f)
        write.writerows(writeToTestFile)
        print('writeToFile')
        pprint(writeToFile)
        print('writeToTestFile')
        pprint(writeToTestFile)


def write_to_csv(trainOrTest, label, contents):
    arrayToDump = np.insert(contents, 0, label)
    pprint(arrayToDump)
    arrayToDump = arrayToDump.reshape((1, 785))

    dataFileName = ""
    if trainOrTest:
        dataFileName = "train.csv"
    else:
        dataFileName = "test.csv"

    with open(dataFileName, "a") as f:
        np.savetxt(f, arrayToDump, fmt="%d", delimiter=",")


def open_image(pathToFileIndex, trainOrTest):
    print('open_image читаем '+pathToFileIndex)

    with open(pathToFileIndex, "r") as f:
        fileList = csv.reader(f)
        print(fileList)

        for file in fileList:
            fileName = "".join(file)
            print(fileName)
            print(fileName)
            label = fileName[(fileName.rfind("/") - 1)]
            print(label)
            image = Image.open(fileName)

            numpydata = np.asarray(image)
            print(numpydata)
            print(numpydata[0])
            # Загрузим из датасета определенное фото и отобразим его на экране. Должна отобразиться цифра, которая находится на позиции photo_number.
            print("Number:" )
            plt.imshow(numpydata, cmap='gray')
            plt.show()  #формат данных такой же, как и в my.py
            write_to_csv(trainOrTest, label, numpydata)
            exit(-1)


file_indexer()
open_image("Files.csv", True)
open_image("TestFiles.csv", False)