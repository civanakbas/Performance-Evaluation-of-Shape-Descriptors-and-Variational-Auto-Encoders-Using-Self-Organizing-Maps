import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from som import SOM

colors = np.empty((0, 3), float)

for i in range(10):
    colors = np.append(colors, np.array([[random.random(), random.random(), random.random()]]), axis=0)

data = torch.Tensor(colors)

row = 20
col = 20
total_epoch = 1500

som = SOM(3, (row, col))

for epoch in range(total_epoch):
    som.self_organizing(data, epoch, total_epoch)
    print(f"Epoch: {epoch}")

# print(som.weight)
weight = som.weight.reshape(3, row, col).detach().numpy()

weight = np.transpose(weight, (1, 2, 0))
plt.imshow(weight)
plt.show()
