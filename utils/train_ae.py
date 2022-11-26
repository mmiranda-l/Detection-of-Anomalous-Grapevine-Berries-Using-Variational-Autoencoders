import time

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torch
from models.ae import SegNet
from utils import utils
import os


from utils.data import get_data_loader
from models.ae import VAE
from utils import config
from PIL import Image

logdir = "./log/vae-123"
batch_train = 64
batch_test = 16
epochs = 120
gpu = "0" 
initial_lr = 0.0005
alpha = 1.0
beta = 0.5 
model = "vae-123"


l1_loss = nn.L1Loss(reduce=False)



def train(epoch, train_loader, model, optimizer):
    # Ensure dropout layers are in train mode
    model.train()

    # Loss function
    # criterion = nn.MSELoss().to(device)

    batch_time = utils.ExpoAverageMeter()  # forward prop. + back prop. time
    losses = utils.ExpoAverageMeter()  # loss (per word decoded)

    start = time.time()

    # Batches
    for i_batch, (x, y) in enumerate(train_loader):
        # Set device options
        x = x.to(config.device)
        y = y.to(config.device)

        # print('x.size(): ' + str(x.size())) # [32, 3, 224, 224]
        # print('y.size(): ' + str(y.size())) # [32, 3, 224, 224]

        # Zero gradients
        optimizer.zero_grad()

        y_hat = model(x)
        # print('y_hat.size(): ' + str(y_hat.size())) # [32, 3, 224, 224]

        #loss = torch.sqrt((y_hat - y).pow(2).mean())
        loss = l1_loss(y_hat, y)
        loss.backward()

        # def closure():
        #     optimizer.zero_grad()
        #     y_hat = model(x)
        #     loss = torch.sqrt((y_hat - y).pow(2).mean())
        #     loss.backward()
        #     losses.update(loss.item())
        #     return loss

        # optimizer.step(closure)
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i_batch % 2 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i_batch, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  loss=losses))


def valid(val_loader, model):
    model.eval()  # eval mode (no dropout or batchnorm)

    # Loss function
    # criterion = nn.MSELoss().to(device)

    batch_time = utils.ExpoAverageMeter()  # forward prop. + back prop. time
    losses = utils.ExpoAverageMeter()  # loss (per word decoded)

    start = time.time()

    with torch.no_grad():
        # Batches
        for i_batch, (x, y) in enumerate(val_loader):
            # Set device options
            x = x.to(config.device)
            y = y.to(config.device)

            y_hat = model(x)

            loss = torch.sqrt((y_hat - y).pow(2).mean())

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i_batch % 2 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i_batch, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))

    return losses.avg


def main():
    for epoch in range(epochs):
        for phase in ["train", "test"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)
            # Loss
            running_loss = 0.0
            # Data num
            data_num = 0

            # Train
            for i, x in enumerate(dataloaders[phase]):
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

def main():
    #train_loader = DataLoader(dataset=VaeDataset('train'), batch_size=batch_size, shuffle=True,
    #                          pin_memory=True, drop_last=True)
    #val_loader = DataLoader(dataset=VaeDataset('valid'), batch_size=batch_size, shuffle=False,
    #                        pin_memory=True, drop_last=True)

    dataloaders = get_data_loader(batch_train, batch_test)

    # Create SegNet model
    label_nbr = 3
    model = SegNet(label_nbr)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [40, xxx] -> [10, ...], [10, ...], [10, ...], [10, ...] on 4 GPUs
        model = nn.DataParallel(model)
    # Use appropriate device
    model = model.to(config.device)
    # print(model)

    # define the optimizer
    # optimizer = optim.LBFGS(model.parameters(), lr=0.8)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = 100000
    epochs_since_improvement = 0

    # Epochs
    for epoch in range(0, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(epoch, train_loader, model, optimizer)

        # One epoch's validation
        val_loss = valid(val_loader, model)
        print('\n * LOSS - {loss:.3f}\n'.format(loss=val_loss))

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, val_loss, is_best)


if __name__ == '__main__':
    main()