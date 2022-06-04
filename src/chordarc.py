import torch
import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import glob
import cv2
import re
from pathlib import Path
import math
from som import SOM
from algorithms import Algorithms
import pickle
from sklearn import preprocessing

Alg = Algorithms()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

DATA_DIR = "../dataset/cut/"
batch_size = 32
total_epoch = 100
train = True

# Size of the output map
out_size = (17, 17)

file_pattern = re.compile(r".*?(\d+).*?")


def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])


image_list = []
for filename in sorted(glob.glob(DATA_DIR + "/*.png"), key=get_order):
    im = cv2.imread(filename)
    image_list.append(im)

chord_arch_list = []
for img in image_list:
    polys = Alg.get_all_poly_points(img)
    chord_arch_list.append(Alg.get_chord_arc(polys))

chord_arch_list = preprocessing.normalize(chord_arch_list)

transform = transforms.ToTensor()
chord_arch_list = torch.Tensor(chord_arch_list)

train_loader = DataLoader(chord_arch_list, batch_size=batch_size, shuffle=True)


som = SOM(input_size=7 * 1, out_size=out_size)
som = som.to(device)


if train is True:
    losses = []
    for epoch in range(total_epoch):
        running_loss = 0
        start_time = time.time()
        for idx, (X) in enumerate(train_loader):
            X = X.view(-1, 7 * 1).to(device)
            loss = som.self_organizing(X, epoch, total_epoch)
            running_loss += loss

        losses.append(running_loss)

        if epoch % 10 == 0:
            print(
                "epoch = %d, loss = %.2f, time = %.2fs"
                % (epoch + 1, running_loss, time.time() - start_time)
            )
        plt.plot(losses)


bmu_locations, _ = som.forward(chord_arch_list.to(device))
bmu_locations = (
    bmu_locations.to("cpu")
    .detach()
    .numpy()
    .astype(int)
    .reshape(bmu_locations.shape[0], bmu_locations.shape[2])
)
print(bmu_locations)
pickle.dump(bmu_locations, open("../bmu_locations/bmu_locations_ca.pkl", "wb"))

plt.show()
