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

'''
photo_number=256 #Загрузим из датасета определенное фото и отобразим его на экране. Должна отобразиться цифра, которая находится на позиции photo_number.
print ("Number:", data_orig["train"]["y"][photo_number])
plt.imshow(data_orig["train"]["x"][photo_number], cmap='gray')
plt.show()
'''

def get_vector_by_number(count):
    r"""
    Преобразует число в списке в выходной вектор
    Например:
    1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    """

    def f(number):
        res = [0.0] * count

        if (number >= 0 and number < count):
            res[number] = 1.0

        return res

    return f


def data_normalize_x(data_x):
    r"""
    Нормализация датасета по x
    """
    data_x = torch.from_numpy(data_x)
    data_x_shape_len = len(data_x.shape)

    if data_x_shape_len == 3:
        data_x = data_x.reshape(data_x.shape[0], -1)
    elif data_x_shape_len == 2:
        data_x = data_x.reshape(-1)

    data_x = data_x.to(torch.float32) / 255.0
    return data_x


def data_normalize_y(data_y):
    r"""
    Нормализация датасета по y
    """
    data_y = list(map(get_vector_by_number(10), data_y))
    data_y = torch.tensor(data_y)
    return data_y

# Создание и обучение нейронной сети
# Создадим нормализованный датасет:
batch_size = 128

data = {
  "train": {
    "x": data_normalize_x(data_orig["train"]["x"]),
    "y": data_normalize_y(data_orig["train"]["y"]),
  },
  "test": {
    "x": data_normalize_x(data_orig["test"]["x"]),
    "y": data_normalize_y(data_orig["test"]["y"]),
  }
}

train_dataset = TensorDataset( data["train"]["x"], data["train"]["y"] )
test_dataset = TensorDataset( data["test"]["x"], data["test"]["y"] )

train_count = data["train"]["x"].shape[0]
test_count = data["test"]["x"].shape[0]

train_loader = DataLoader(
  train_dataset,
  batch_size=batch_size,
  drop_last=True,
  shuffle=True
)
test_loader = DataLoader(
  test_dataset,
  batch_size=batch_size,
  drop_last=True,
  shuffle=False
)

# Архитектура модели:
def create_model():
  input_shape = 784
  output_shape = 10

  model = nn.Sequential(
    nn.Linear(input_shape, 128),
    nn.ReLU(),
    nn.Linear(128, output_shape),
    #nn.Softmax()
  )

  # Adam optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

  # mean squared error
  loss = nn.MSELoss()

  return {
		"input_shape": input_shape,
		"output_shape": output_shape,
		"model": model,
		"optimizer": optimizer,
		"loss": loss,
	}


# Выведем информацию о модели на экран:

model_info = create_model()

# Show model info
summary(model_info["model"], (model_info["input_shape"],))