import matplotlib.pyplot as plt
from PIL import Image
import glob
import pickle
import re
from pathlib import Path
import math

DATA_DIR = "dataset/contours/"

file_pattern = re.compile(r".*?(\d+).*?")


def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])


contour_list = []
for filename in sorted(glob.glob(DATA_DIR + "*.png"), key=get_order):
    im = Image.open(filename)
    contour_list.append(im)


labels = pickle.load(open("./labels/contours/labels.pkl", "rb"))
print(labels)
i = 0
while i < len(contour_list):
    if i == 0:
        label_from_checkpoint = input(
            f"Currenty labeled objects: {len(labels)}, continue labeling from this point? True/False\n"
        )
        if label_from_checkpoint == "True":
            i += len(labels)

    plt.imshow(contour_list[i])
    plt.title(f"obj_{i}")
    plt.show()
    label = input(
        "Please enter this images label [elongated, round, meander, crossed]: Type exit to stop labeling\n"
    )
    if label == "exit":
        break
    labels[f"obj_{i}"] = label
    i += 1

pickle.dump(labels, open("./labels/contours/labels.pkl", "wb"))
print(labels)
