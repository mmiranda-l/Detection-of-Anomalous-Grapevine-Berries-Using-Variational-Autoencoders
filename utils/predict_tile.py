import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import cv2


from torch.utils.data import DataLoader


import matplotlib
import matplotlib.image as mpimg
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(".")

from utils.anno import gen_images
from utils.vis import imsave_inp
from models.vae import VAE
import config
from utils.vis import plot_loss, imsave, Logger, imsave_inp
from sklearn.manifold import Isomap, MDS
from PIL import Image
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from utils import data

from sklearn.svm import OneClassSVM

df = pd.DataFrame(columns = [str(i) for i in range(130)])

path = ""

image_path = ""
mask_path = ""

base_img = np.array(Image.open(image_path).convert("RGB"))
base_copy = np.array(base_img.copy())

def mds(feature_values, n_components: int = 2):
    """Multi-dimensional scaling"""
    embeding = MDS(n_components=n_components).fit_transform(feature_values)
    mds_df = pd.DataFrame(data = embeding
                 , columns = ['mds'+str(i+1) for i in range(n_components)])
    return mds_df


latent_df = pd.read_csv(os.path.join(path,"latent_space.csv"))

mds_df = latent_df[list(latent_df.columns[:-1])]
mds_df["class"] = latent_df["128"]
X = mds_df[mds_df.columns[:-1]]
Y = mds_df["class"]
Y = Y.replace("healthy", 1)
Y = Y.replace("damaged", 2)
X_train, X_test, y_train, y_test = train_test_split(
     X, Y, test_size=0.33, random_state=42)
print(len(X_train.columns))

clf = SVC().fit(X_train,y_train)
pred = clf.predict(X_test)
print(accuracy_score(y_test, pred))

random = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
pred = clf.predict(X_test)
print(accuracy_score(y_test, pred))

print("trained classifier")
#print(accuracy_score([-1 for i in range(len(pred))], pred))

model = VAE(device=config.DEVICE, is_train=False)
model.load_state_dict(torch.load(os.path.join(path, "latest.pth"), map_location=torch.device(config.DEVICE)))


datset = data.TileDataSet(image_path, mask_path, is_aug=False)
dataloaders = DataLoader(datset, batch_size=2, shuffle=True)

h = 0
with torch.no_grad():
    for _, data_dict in enumerate(dataloaders):
        input = data_dict["image"]
        imsave_inp(input,"log/image{}.png".format(_) )
        x = model.set_input(input)
        z, rec_x, mean, logvar = model(input)
        for index, latent in enumerate(z):
            print(len(latent))
            i,j = int(data_dict["x"][index]), int(data_dict["y"][index])
            detached = list(latent.cpu().detach().numpy())            
            detached.append(i)
            detached.append(j)
            df.loc[h] = detached
            h += 1
#df.to_csv("test.csv")
print('save Data Frame')


#mds_df = mds(df[df.columns[:-2]], n_components=40)

input_df = df[df.columns[:-2]]
print(len(input_df.columns))
print("reduced data dimensions")
classes = []
for i in range(len(input_df)):
    x = int(df.iloc[i]["128"])
    y = int(df.iloc[i]["129"])
    input = input_df.iloc[i]
    svm_class = clf.predict(np.array(input).reshape(1, -1))
    rfl_class = random.predict(np.array(input).reshape(1, -1))
    classes.append(rfl_class)
    try: 
        if rfl_class == 1:
            base_copy[y:y + 130, x:x + 130] = base_img[y:y + 130, x:x + 130]
            pass
        else:
            base_copy[y:y + 130, x:x + 130] =  np.array(Image.new('RGB', (130, 130), "red"))
            
    except:pass
print(np.unique(classes))
plt.imsave("test.png", base_copy)

