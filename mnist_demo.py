import torch
import time
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from som import SOM
import os
import argparse




parser = argparse.ArgumentParser(description="Self Organizing Map Implementation/ Demo on MNIST")
parser.add_argument("--batch_size", type=int, default=32 ,help="Set the batch size")
parser.add_argument("--lr", type=float, default=0.3, help="Set the learning rate")
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--result_dir", type=str, default="results", help="Destionation folder for generated maps")
parser.add_argument("--train", type=bool, default=True)
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

DATA_DIR = "datasets/mnist"
RESULT_DIR = args.result_dir
batch_size = args.batch_size
lr = args.lr
total_epoch = args.epoch
train=args.train

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)




# Size of the output map
out_size=(40,40)

transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
train_data.data = train_data.data[:50]
train_data.targets = train_data.targets[:50]
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

som = SOM(input_size= 28 * 28 * 1, out_size=out_size)

som = som.to(device)


if train is True:
    losses = list()
    for epoch in range(total_epoch):
        running_loss = 0
        start_time = time.time()
        for idx, (X,Y) in enumerate(train_loader):
            X = X.view(-1, 28 * 28 * 1).to(device)
            loss = som.self_organizing(X, epoch, total_epoch)
            running_loss += loss

        losses.append(running_loss)
        print(
            "epoch = %d, loss = %.2f, time = %.2fs"
            % (epoch + 1, running_loss, time.time() - start_time)
        )

        if epoch % 5 == 0:
            som.save_result("%s/som_epoch_%d.png" % (RESULT_DIR, epoch), (1, 28, 28))
        plt.plot(losses)


    som.save_result("%s/som_result.png" % RESULT_DIR, (1,28,28))
plt.show()
