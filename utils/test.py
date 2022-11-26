import os
import sys
import argparse
import numpy as np
import pandas as pd
from pyrsistent import rex
import torch
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
import torch.nn as nn
import cv2
import tifffile as tiff
from PIL import Image

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
from utils.loss import FLPLoss

from utils import data
import cv2

#parser = argparse.ArgumentParser(description='vae.pytorch')
#parser.add_argument('--logdir', type=str, default="./log/vae-123")
#parser.add_argument('--num', type=int, default=10)
#parser.add_argument('--gpu', type=str, default="0")
#parser.add_argument('--model', type=str, default="vae-123", choices=["vae-123", "vae-345", "pvae"])
#parser.add_argument('--path', type=str, default="./log/vae-123/final_model.pth")
#parser.add_argument('--attr', type=str, default="Smiling")
#args = parser.parse_args()

logdir = "./log/vae-123"
batch_train = 64
batch_test = 16
epochs = 800
gpu = "0" 
initial_lr = 0.0005
alpha = 1.0
beta = 0.5 
model = "vae-123"
path = ""

result_path = ""
num = 10

data_root = ""

logdir = os.path.join(logdir, "prediction")
if not os.path.exists(logdir):
    os.makedirs(logdir)

model = VAE(device=config.DEVICE, is_train=False)

model.load_state_dict(torch.load(path, map_location=torch.device(config.DEVICE)))

dataloaders = data.get_data_loader_test(25)
columns = [str(i) for i in range(128)]
columns = columns.append("class")
df = pd.DataFrame(columns = [str(i) for i in range(129)])
j = 0

l1_loss = nn.L1Loss(reduce=False)

mse_loss = nn.MSELoss(reduce=False)
bce_loss = nn.BCELoss(reduce=False)

def get_binary_map(loss_map, img, threshold=0.45, kernel = np.ones((3,3),np.uint8)):
    binary_map = Image.new(mode="RGBA", size=(loss_map.shape[0],loss_map.shape[1]))
    smallest = loss_map.min(axis=(0, 1))

    largest = loss_map.max(axis=(0, 1))
    for x in range(binary_map.width):
        for y in range(binary_map.height):
            value = ((loss_map[y,x] - smallest)*(255/(largest-smallest)))
            if loss_map[y,x] > threshold: 
                binary_map.putpixel((x,y),(0,0,int(value), 255))
            else: 
                binary_map.putpixel((x,y),(0,0,0, 0))
            pixel = img[y,x]
            pixel = [pixel[0]*value, pixel[1], pixel[2]]
            img[y,x] = pixel 
    binary_map = np.array(binary_map)
    opening = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
    na = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return na, img

with open(result_path, "w") as file: 

    with torch.no_grad():
        for phase in ["healthy", "damaged"]:
            for i, (x, path) in enumerate(dataloaders[phase]):
                if phase == "healthy":
                   original_path = os.path.join(config.TEST_PATH_HEALTHY)
                elif phase == "damaged":
                   original_path = os.path.join(config.TEST_PATH_HEALTHY)
                x = model.set_input(x)
                z, rec_x, mean, logvar = model(x)
                loss = model.get_losses(x, rec_x, mean, logvar) 
                kl = model.get_kl(x, rec_x, mean, logvar)
                rec_loss = model.get_log_loss(x, rec_x, mean, logvar)
                pixelwise_loss = l1_loss(x, rec_x)
                loss = model.bce(x, rec_x)                
                for index, l in enumerate(pixelwise_loss):      
                    pass
                    file.write( str(float(loss)) + "\t"+ str(float(kl))+ "\t"+ str(float(rec_loss)) + 
                    "\t" +  str(float(loss)) +  "\t"+ str(float(loss)) + "\t" +  phase + "\n")              
                    loss_img = pixelwise_loss.clone().cpu().detach().numpy().astype(np.float64).transpose(0, 2, 3, 1)[:, :, :, ::-1][index]
                    stacked = loss_img[:,:,0] + loss_img[:,:,1] + loss_img[:,:,2]

                    stacked_int = (stacked*200).astype(np.uint8)
                  
                    img = imsave_inp(x, logdir+"test_img.png")[index]
                    rec_img = imsave_inp(rec_x, logdir+"test_img.png")[index]
                    fig, axs = plt.subplots(1,2, figsize=(16,10))
                    tiff.imsave(os.path.join(logdir, path[index]+'_loss_{}.tiff'.format(phase)), cv2.resize(stacked, (130,130)))
                    binary, img = get_binary_map(stacked, img)
                    cv2.imwrite(os.path.join(logdir, path[index]+'_img_binary_{}.png'.format(phase)), cv2.resize(binary, (130,130)))
                    cv2.imwrite(os.path.join(logdir, path[index]+'_img_img_{}.png'.format(phase)), cv2.resize(img, (130,130)))
                    plt.figure(figsize=(5,5))
                    ax=plt.subplot(111)
                    sns.heatmap(stacked, cmap = sns.diverging_palette(220, 20, as_cmap=True),ax=ax, annot=False, cbar=False, xticklabels=False, yticklabels=False)
                    plt.savefig(os.path.join(logdir, path[index]+"_HeatMapOnly_{}_{}.png".format(phase, i)))
                    plt.close()

                for i in z:
                    liste = list(i.cpu().detach().numpy())
                    liste.append(phase)
                    df.loc[j] = liste
                    j+=1
               






