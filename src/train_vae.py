from vae import VAE
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import glob
import cv2
import re
from pathlib import Path
import math
from torch.utils.data import DataLoader


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


train_dataset = torch.Tensor(contour_list)

image_size = 100 * 100
batch_size = 8
epochs = 2000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model = VAE(image_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def loss_fn(reconstructed_image, original_image, mu, log_var):
    bce = F.binary_cross_entropy(
        reconstructed_image, original_image.view(-1, image_size), reduction="sum"
    )
    kld = 0.5 * torch.sum(log_var.exp() + mu.pow(2) - 1 - log_var)
    return bce + kld


def train(epoch):
    model.train()
    train_loss = 0
    for images in train_loader:
        images = images.to(device)
        reconstructed, mu, log_var = model(images)
        loss = loss_fn(reconstructed, images, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader.dataset)
    print(f"=====> Epoch {epoch}, Average Loss: {avg_loss:.2f}")

    return avg_loss


if not os.path.exists("./savedmodel"):
    os.makedirs("./savedmodel")

running_loss = []
min_loss = 2000
for epoch in range(1, epochs + 1):
    curr_loss = train(epoch)
    running_loss.append(curr_loss)
    if curr_loss <= min_loss:
        min_loss = curr_loss
        torch.save(model, "./savedmodel/vae_model.pt")
        print(
            f"====->>>>>Saved the model at {epoch} epoch at {min_loss:.2f} Loss Value :)"
        )

plt.plot(running_loss)
plt.title(f"Average loss for {epochs} epoch")
plt.show()

print(f"====->>>>>Lastly saved the model at {min_loss:.2f} Loss Value :)")
