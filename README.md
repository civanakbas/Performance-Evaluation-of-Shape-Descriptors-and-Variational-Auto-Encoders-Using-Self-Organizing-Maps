# Performance Evaluation of Shape Descriptors and Variational Auto-Encoders Using Self Organizing Maps
This repo contains the codes done for the CEN438 Graduation Thesis course.

## Requirements
```
python==3.6
pytorch===1.1.0
torchvision==0.3.0
opencv==4.5.5.64
scipy
scikit-learn
pickle
numpy
matplotlib
```

There are 2 simple demos for Self Organizing Maps. One for MNIST and one for random colors.
```
python mnist_demo.py
python color_demo.py
```
For MNIST demo generated images are in ./result folder by default.

## Roughly Visualize Maps
To roughly visualize the output maps use `python visualize_clusters.py --descriptor cch/pgh/cbsd/ca/vae`

### Generate New Best Matching Units
Best maching units for each method is calculated and saved using `pgh.py` `cch.py` `cbsd.py` `ca.py` `encode_dataset_with_vae.py` and saved as a pickle in `bmu_locations` folder. These scripts can be used to generate new BMU's for each method.
