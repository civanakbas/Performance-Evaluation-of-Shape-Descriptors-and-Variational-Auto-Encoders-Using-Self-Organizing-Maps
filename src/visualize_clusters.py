import pickle
from tkinter import Label, Tk
from PIL import Image, ImageTk
import argparse

parser = argparse.ArgumentParser(
    description="Visualize clusters using Best Maching Units calculated"
)
parser.add_argument(
    "--descriptor",
    type=str,
    help="Set the descriptor (cch,pgh,cbsd,ca,vae)",
    required=True,
)
args = parser.parse_args()
descriptor = args.descriptor


bmu_locations = pickle.load(
    open("../bmu_locations/bmu_locations_" + descriptor + ".pkl", "rb")
)

args = parser.parse_args()

root = Tk()
root.configure(bg="white")

for i, (x, y) in enumerate(bmu_locations):
    j = i + 1
    while j < len(bmu_locations):
        if (x, y) == (bmu_locations[j][0], bmu_locations[j][1]):
            print(f"obj{i+1} returns the same BMU as obj{j+1}")
        j += 1

labels = [Label] * 153
for i in range(153):
    image = Image.open("../dataset/contours/obj_" + str(i + 1) + ".png")
    image = image.resize((50, 50))
    image = ImageTk.PhotoImage(image)
    labels[i] = Label(image=image)
    labels[i].config(borderwidth=0)
    labels[i].image = image

for i, _ in enumerate(labels):
    labels[i].grid(row=(bmu_locations[i][0]), column=(bmu_locations[i][1]))

root.mainloop()
