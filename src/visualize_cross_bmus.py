# Dünyanın en kötü kodu.
import os
import pickle
import cv2 as cv
from matplotlib import pyplot as plt

def get_dict(loc):
    bmu_locations = pickle.load(
        open(f"../bmu_locations/bmu_locations_{loc}.pkl","rb"))
    dict = {}
    for i,(x,y) in enumerate(bmu_locations):
        dict[f"{x},{y}"] = []
    for i,(x,y) in enumerate(bmu_locations):
        dict[f"{x},{y}"].append(f"obj_{i+1}")
    one_values = []
    for key,val in dict.items():
        if len(val) == 1:
            one_values.append(key)
    for key in one_values:
        dict.pop(key)
    
    return dict

def save_figs(dict,loc):
    dir = f"../results/{loc}"
    for f in os.listdir(dir):
        f = os.path.join(dir,f)
        # print(f)
        # return
        os.remove(f)
    for key in dict:
        N = len(dict[key])
        objs = dict[key]
        fig, axs = plt.subplots(1,N)
        fig.suptitle(str(key))
        for i in range(N):
            img = cv.imread(f"../dataset/contours/{objs[i]}.png")
            axs[i].imshow(img)
            axs[i].set_title(str(objs[i]))
            axs[i].set_yticks([])
            axs[i].set_xticks([])
        plt.savefig(f"../results/{loc}/{key}.png")
        plt.close()


dict = get_dict("pgh")
save_figs(dict,"pgh")
dict = get_dict("ca")
save_figs(dict,"ca")
dict = get_dict("cbsd")
save_figs(dict,"cbsd")
dict = get_dict("cch")
save_figs(dict,"cch")
dict = get_dict("vae")
save_figs(dict,"vae")






