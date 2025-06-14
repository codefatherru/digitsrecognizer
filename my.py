import torch, math
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset

tensor_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print ("Device:", tensor_device)

data_orig = np.load("mnist.npz", allow_pickle=True)
data_orig = {
  "train": {
    "x": data_orig["x_train"],
    "y": data_orig["y_train"],
  },
  "test": {
    "x": data_orig["x_test"],
    "y": data_orig["y_test"],
  }
}

print ("Train images", data_orig["train"]["x"].shape)
print ("Train answers", data_orig["train"]["y"].shape)
print ("Test images", data_orig["test"]["x"].shape)
print ("Test answers", data_orig["test"]["y"].shape)

photo_number=256 #Загрузим из датасета определенное фото и отобразим его на экране. Должна отобразиться цифра, которая находится на позиции photo_number.
print ("Number:", data_orig["train"]["y"][photo_number])
plt.imshow(data_orig["train"]["x"][photo_number], cmap='gray')
plt.show()