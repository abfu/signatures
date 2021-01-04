"""
For now focus on wether two signatures are matches or not.

Afterwards add forgery detection to network.

"""
import torch.nn as nn
import torchvision.datasets as dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import pandas as pd
from explore import extract, compare_pic
from utils import SignatureDataset, Rescale, ToTensor
import matplotlib.pyplot as plt

DATADIR = './data/sign_data/'

rescale = Rescale((100, 100))
train_set = SignatureDataset(root='./data/sign_data/',
                             split='train',
                             transforms=transforms.Compose([rescale,
                                                            ToTensor()]))

train_loader = DataLoader(train_set, batch_size=50, shuffle=True)

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched['img_a'].size(),
          sample_batched['is_match'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        break

