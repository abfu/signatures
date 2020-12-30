import torch.nn as nn
import torchvision.datasets as dataset
from torchvision.transforms import Resize, Compose
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from explore import extract, compare_pic


DATADIR = './data/sign_data/'

train = extract(DATADIR, 'train')
test = extract(DATADIR, 'test')

class SignatureDataset(Dataset):
    def __init__(self, dir):
        self.file =
