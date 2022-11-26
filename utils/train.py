import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append(".")

from utils.data import get_data_loader
from utils.vis import plot_loss, imsave, Logger
from utils.loss import FLPLoss, KLDLoss
from models.vae import VAE
from pytorch_msssim import ssim

from utils import config


logdir = "./log/vae-123"
batch_train = 64
batch_test = 16
epochs = 120
gpu = "0" 
initial_lr = 0.0005
alpha = 1.0
beta = 0.5 
model = "vae-123"


# Set GPU (Single GPU usage is only supported so far)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

# Dataloader
dataloaders = get_data_loader(batch_train, batch_test)
# Model
model = VAE(device=config.DEVICE).to(config.DEVICE)

# Reconstruction loss
#if model == "pvae":
#    reconst_criterion = nn.MSELoss(reduction='sum')
#elif model == "vae-123" or model == "vae-345":

#OLD LOSSES
#reconst_criterion = FLPLoss(model, device, reduction='sum')
# KLD loss
kld_criterion = KLDLoss(reduction='sum')


# Solver
# Scheduler
scheduler = optim.lr_scheduler.StepLR(model.optimizer, 1, gamma=0.5, last_epoch=-1)
# Log
logdir = logdir
if not os.path.exists(logdir):
    os.makedirs(logdir)
# Logger
logger = Logger(os.path.join(logdir, "log.txt"))
# History
history = {"train": [], "test": []}

# Save config
logger.write('----- Options ------')

# Start training

for epoch in range(epochs):
    for phase in ["train", "test"]:
        if phase == "train":
            model.train(True)
            logger.write(f"\n----- Epoch {epoch+1} -----")
        else:
            model.train(False)
        # Loss
        running_loss = 0.0
        # Data num
        data_num = 0
        # Train
        for i, (x, path) in enumerate(dataloaders[phase]):
            # Optimize params
            if phase == "train":
                model.optimizer.zero_grad()

                # Pass forward
                x = model.set_input(x)
                _, rec_x, mean, logvar = model(x)

                # Calc loss
                #reconst_loss = reconst_criterion(x, rec_x)
                loss = model.get_losses(x, rec_x, mean, logvar)
                #loss = kld_loss + l1 
                model.update(loss)
                #loss.backward()
                #model.optimizer.step()

                # Visualize
                if i == 0 and x.size(0) >= 64:
                    imsave(x, rec_x, os.path.join(logdir, f"epoch{epoch+1}", f"train.png"), 8, 8)


            elif phase == "test":
                with torch.no_grad():
                    #model.optimizer.zero_grad()
                    # Pass forward
                    x = model.set_input(x)
                    _, rec_x, mean, logvar = model(x)

                    # Calc loss
                    loss = model.get_losses(x, rec_x, mean, logvar) 

                    # Visualize
                    if x.size(0) >= 16:
                        imsave(x, rec_x, os.path.join(logdir, f"epoch{epoch+1}", f"test-{i}.png"), 4, 4)

            # Add stats
            running_loss += loss # * x.size(0)
            data_num += x.size(0)

        # Log
        epoch_loss = running_loss / data_num
        logger.write(f"{phase} Loss : {epoch_loss:.4f}")
        history[phase].append(epoch_loss)

        if phase == "test":
           # plot_loss(logdir, history)
            pass
    if epoch % 5 == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch {}'.format(epoch))
            torch.save(model.state_dict(),\
            os.path.join(logdir, 'latest.pth'))

# Save the model
torch.save(model.state_dict(),\
    os.path.join(logdir, 'final_model.pth'))