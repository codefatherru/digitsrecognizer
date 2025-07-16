# пробный файл для преобразования изображений к MNIST

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import PIL
import numpy as np
import cv2


png = "dataset\\val\\viii\\8_cap_450.png"
png = "dataset\\train\\viii\\viii_031.png"

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

    # применяем пороговую обработку
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    gray = cv2.resize(gray, (28, 28))
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