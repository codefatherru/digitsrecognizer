# пробный файл для преобразования изображений к MNIST
import math

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import PIL
import numpy as np
import cv2


png = "dataset\\val\\viii\\8_cap_450.png"
png = "dataset\\train\\viii\\viii_031.png"
png = "dataset\\train\\viii\\viii_401.png"

# loads the images in grayscale mode and converts all the pixels that aren’t very dark (brightness of 43 or less) to white
def convert_images(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    input_files = glob(os.path.join(input_folder, "*.png"))
    for f in input_files:
        image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        # quantize
        image = (image // 43) * 43
        image[image > 43] = 255
        cv2.imwrite(os.path.join(output_folder, os.path.basename(f)), image)



def rec_digit(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gray = 255 - img
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

    cv2.imwrite('gray_box.png', gray)

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

    cv2.imwrite('gray_20.png', gray)

    #расширяем картинку до 28x28 пикселей, добавляя черные ряды и столбцы по краям
    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.pad(gray, (rowsPadding, colsPadding), 'constant')

    cv2.imwrite('gray.png', gray)
    #img = gray / 255.0
    #img = np.array(img).reshape(-1, 28, 28, 1)
    #out = str(np.argmax(model.predict(img)))
    return gray #массив 28на28


image_file = Image.open(png)
inverted_img = ImageOps.invert(image_file)
img_grey = inverted_img.convert('L') # convert image to grayscale
#image_file.save('testimage.gif')

value = np.asarray(img_grey.getdata(), dtype=np.int32 ).reshape((img_grey.size[1], img_grey.size[0]))
#print(value)
# перевели матрицу в массив
fvalue = value.flatten()
#print(fvalue)

#отобразим матрицу
plt.imshow(value, cmap='gray')
plt.show()

out = rec_digit(png)
print(out)