import torch
import re
import glob
import cv2
import math
from pathlib import Path

import numpy as np
import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import math
import pickle

model = torch.load("./savedmodel/vae_model.pt")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

file_pattern = re.compile(r".*?(\d+).*?")


def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])


DATA_DIR = "../dataset/contours"
contour_list = []
for filename in sorted(glob.glob(DATA_DIR + "/*.png"), key=get_order):
    im = cv2.imread(filename)
    raw_gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    th, bin_img = cv2.threshold(raw_gray_img, 127, 255, cv2.THRESH_OTSU)
    bin_img = bin_img / 255
    im = bin_img.reshape(1, 100, 100)
    contour_list.append(im)

image_dataset = torch.Tensor(contour_list)

image_size = 100 * 100
batch_size = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    dataset=image_dataset, batch_size=batch_size, shuffle=False
)

encoded_with_batch = []
with torch.no_grad():
    for i, images in enumerate(train_loader):
        images = images.to(device)
        mu, log_var = model.encode(images.view(-1, 100 * 100))
        enc = model.reparameterize(mu, log_var)
        encoded_with_batch.append(enc)


def unbatch(tensor):
    unbatched_list = []
    for batch in tensor:
        for encoded in batch:
            unbatched_list.append(np.array(encoded.cpu()))
    return unbatched_list


from som import SOM

batch_size = 128
total_epoch = 2000
train = True

out_size = (9, 9)
transform = transforms.ToTensor()
vae_encoded_list = torch.Tensor(unbatch(encoded_with_batch))
vae_encoded_list = vae_encoded_list.cpu()

train_loader = DataLoader(vae_encoded_list, batch_size=batch_size, shuffle=False)
som = SOM(input_size=20 * 1, out_size=out_size)
som = som.to(device)

if train is True:
    losses = []
    for epoch in range(total_epoch):
        running_loss = 0
        start_time = time.time()
        for idx, (X) in enumerate(train_loader):
            X = X.view(-1, 20 * 1).to(device)
            loss = som.self_organizing(X, epoch, total_epoch)
            running_loss += loss

        losses.append(running_loss)

        if epoch % 10 == 0:
            print(
                "epoch = %d, loss = %.2f, time = %.2fs"
                % (epoch + 1, running_loss, time.time() - start_time)
            )
        plt.plot(losses)

print(vae_encoded_list)
print(type(vae_encoded_list))
bmu_locations, _ = som.forward(vae_encoded_list.to(device))
bmu_locations = (
    bmu_locations.to("cpu")
    .detach()
    .numpy()
    .astype(int)
    .reshape(bmu_locations.shape[0], bmu_locations.shape[2])
)

print(bmu_locations)
pickle.dump(bmu_locations, open("../bmu_locations/bmu_locations_vae.pkl", "wb"))
plt.show()
