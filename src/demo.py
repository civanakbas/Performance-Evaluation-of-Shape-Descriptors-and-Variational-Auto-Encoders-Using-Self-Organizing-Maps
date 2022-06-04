import torch
import time
from som import SOM
import os
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(
    description="Self Organizing Map Implementation/ Demo on contours"
)
parser.add_argument("--batch_size", type=int, default=16, help="Set the batch size")
parser.add_argument("--lr", type=float, default=0.3, help="Set the learning rate")
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument(
    "--result_dir",
    type=str,
    default="../results/contour",
    help="Destionation folder for generated maps",
)
parser.add_argument("--train", type=bool, default=True)
args = parser.parse_args()

DATA_DIR = "../dataset/contours"
RESULT_DIR = args.result_dir


if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

args.result_dir
batch_size = args.batch_size
device = "cuda:0" if torch.cuda.is_available() else "cpu"
total_epoch = args.epoch
train = args.train


transform = transforms.ToTensor()
contour_list = []
for filename in glob.glob(DATA_DIR + "/*.png"):
    im = Image.open(filename)
    im = im.resize((30, 30))
    im = transform(im)
    contour_list.append(im)


train_loader = DataLoader(contour_list, batch_size=batch_size, shuffle=True)

som = SOM(input_size=30 * 30 * 1, out_size=(40, 40))
som = som.to(device)


if train is True:
    losses = []
    for epoch in range(total_epoch):
        running_loss = 0
        start_time = time.time()
        for idx, (X) in enumerate(train_loader):
            X = X.view(-1, 30 * 30 * 1).to(device)
            loss = som.self_organizing(X, epoch, total_epoch)
            running_loss += loss

        losses.append(running_loss)
        print(
            "epoch = %d, loss = %.2f, time = %.2fs"
            % (epoch + 1, running_loss, time.time() - start_time)
        )

        if epoch % 5 == 0:
            som.save_result(
                "%s/contour_epoch_%d.png" % (RESULT_DIR, epoch), (1, 30, 30)
            )
        plt.plot(losses)

    som.save_result("%s/contour_result.png" % RESULT_DIR, (1, 30, 30))

plt.show()
