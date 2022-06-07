import torch
import os
import re
import glob
import cv2
import math
from pathlib import Path
from torchvision.utils import save_image

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


RESULT_DIR = "../results/vae_results/contour/"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


train_dataset = torch.Tensor(contour_list)

image_size = 100 * 100
batch_size = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

model.eval()
with torch.no_grad():
    for i, images in enumerate(train_loader):
        images = images.to(device)
        reconstructed, mu, log_var = model(images)

        if i == 9:
            break
        comparsion = torch.cat(
            [images[:5], reconstructed.view(batch_size, 1, 100, 100)[:5]]
        )
        save_image(
            comparsion.cpu(), RESULT_DIR + "reconstruction_" + str(i) + ".png", nrow=5
        )

print(f"Saved the reconstructed images at {RESULT_DIR}/reconstruction*.png")
