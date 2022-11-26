import torch
import torchvision.transforms as transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTENSIONS = ["png", "PNG", "jpg", "JPG", "tiff", "TIFF"]
EPOCHS = 120
BATCH_SIZE = 8
NUM_WORKERS = 4

LEARNING_RATE= 0.0005
LAMBDA_KL = 0.001 #weight for KL loss
LAMBDA_Z = 0.5 # weight for ||E(G(random_z)) - random_z||

BETA = 0.5 


IN_CHANNELS = 3
OUT_CHANNELS = 22
IMG_SIZE = 256

BETA1 = 0.5 #momentum for adam

DATA_PATH = ""
MODEL_PATH = ''
SAVE_PATH =  '' 

TEST_PATH_HEALTHY = r""
TEST_PATH_DAMAGED = r""