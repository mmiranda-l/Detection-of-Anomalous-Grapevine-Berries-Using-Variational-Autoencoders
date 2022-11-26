
from cv2 import reduce
from pytorch_msssim.ssim import ssim
import torch
from torch import nn
import torch.optim as optim

import config
from .networks import Encoder, Decoder
from utils.loss import FLPLoss, KLDLoss
import config

class VAE(nn.Module):
    def __init__(self, in_channels: list=[3, 32, 64, 128, 256], latent_dim: int=128, hidden_dims: list=[32, 64, 128, 256, 512], device="gpu", is_train=True):
        super().__init__()

        self.device = device
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(in_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(hidden_dims, latent_dim)

        self.reconst_criterion = FLPLoss(self, config.DEVICE, reduction='sum')

        self.optimizer = optim.Adam(self.parameters(), lr=config.LEARNING_RATE)
        self.l1 = nn.L1Loss()

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

        
        if is_train == False:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self,x): 
        z, mu, logvar = self.encoder.forward(x)
        decoded = self.decoder.forward(z)
        return z, decoded, mu, logvar

    def set_input(self, x):
        self.optimizer.zero_grad()
        return x.to(self.device)

    def get_losses(self, x, x_rec, mu, logvar, reduction="sum"): 
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        rec_loss = self.reconst_criterion(x, x_rec)
        if reduction == "sum":
            kl_loss = torch.sum(kl_loss)
        elif reduction == "mean":
            kl_loss = torch.mean(kl_loss)
        else:
            kl_loss = 0
        return config.LAMBDA_KL * kl_loss + config.BETA * rec_loss
    
    def get_kl(self, x, x_rec, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        return torch.sum(kl)

    def get_log_loss(self, x, x_rec, mu, logvar):
        return self.reconst_criterion(x, x_rec) 

    def mse_loss(self, x, x_rec):
        return self.l1(x, x_rec), self.mse(x, x_rec)

    def update(self, loss):
        loss.backward() 
        self.optimizer.step()


    

    