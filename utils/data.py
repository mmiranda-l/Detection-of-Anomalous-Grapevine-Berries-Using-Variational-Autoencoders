import os
import numpy as np
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from random import randrange
import random
import numpy as np
import matplotlib.pyplot as plt

import cv2
import config


class ImgAugTransform:
  def __init__(self):
    self.aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        #iaa.Sometimes(0.1,\
        #    iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5)),
        #iaa.Sometimes(0.4,\
        #    iaa.GaussianBlur(sigma=(0, 0.2))),
    ])
      
  def __call__(self, img):
    img = np.array(img)
    img = self.aug.augment_image(img)
    return img

class ImageDataset(Dataset):
    def __init__(self, paths, is_aug=True):
        super(ImageDataset, self).__init__()

        # Length
        self.length = len(paths)
        # Image path
        self.paths = paths
        # Augment
        self.is_aug = is_aug
        self.transform = transforms.Compose([
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ImgAugTransform(),
            lambda x: Image.fromarray(x),
        ])
        # Preprocess
        self.output = transforms.Compose([
            #transforms.Lambda(lambda img: self.__crop_img(img)),
            #transforms.RandomCrop((256, 256)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Image
        path = self.paths[idx]
        patch_name = path.split('/')[-1].split('\\')[-1].split('.')[0]
        img = Image.open(path).convert("RGB")
        # Augment
        if self.is_aug:
            img = self.transform(img)
        # Preprocess
        img = self.output(img)

        return img, patch_name
        #return img 


    def __crop_img(self, img, alpha:int = 0.35): 

        def random_crop(img, size= 128):
            x, y = img.size
            x1 = randrange(0, x - size)
            y1 = randrange(0, y - size)
            return img.crop((x1, y1, x1 + size, y1 + size))

        def pixelratio(img):
            gray_scale = np.array(img.convert("L"))
            count = np.sum(gray_scale == 255)   
            colored =  (img.width * img.height) - count
            return  colored / (img.width * img.height)
        cropped = random_crop(img)
        ratio = pixelratio(cropped)
        if ratio < alpha:
            return self.__crop_img(img)
        else: return cropped

def make_dataset(data_root, subset=1):
    assert os.path.isdir(data_root)
    images = list()
    for file in os.listdir(data_root): 
        if any(file.endswith(extension) for extension in config.EXTENSIONS):
            images.append(os.path.join(data_root, file))

    if subset:
        n_subset = int(len(images) * subset)
        images = random.sample(images, n_subset)
    return images


def get_data_loader(batch_train, batch_test, subset=0.2):
    test_num = 128
    images = make_dataset(config.DATA_PATH)
    datasets = {
        "train": ImageDataset(images[test_num:], True),
        "test": ImageDataset(images[:test_num], False)
    }
    dataloaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_train, shuffle=True),
        "test": DataLoader(datasets["test"], batch_size=batch_test, shuffle=False)
    }

    return dataloaders


def get_data_loader_test(batch_test=2): 
    healthy = make_dataset(config.TEST_PATH_HEALTHY)
    damaged = make_dataset(config.TEST_PATH_DAMAGED)

    datasets = {
        "healthy": ImageDataset(healthy, False),
        "damaged": ImageDataset(damaged, False)
    }
    dataloaders = {
        "healthy": DataLoader(datasets["healthy"], batch_size=batch_test, shuffle=True),
        "damaged": DataLoader(datasets["damaged"], batch_size=batch_test, shuffle=False)
    }

    return dataloaders
