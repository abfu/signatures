"""
For now focus on wether two signatures are matches or not.

Afterwards add forgery detection to network.

"""
import torch
import torch.nn as nn
import torchvision.datasets as dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import pandas as pd
from explore import extract, compare_pic
from utils import SignatureDataset, Rescale, ToTensor
import matplotlib.pyplot as plt
from model import SigNet

DATADIR = './data/sign_data/'

rescale = Rescale((100, 100))
train_set = SignatureDataset(root='./data/sign_data/',
                             split='train',
                             transforms=transforms.Compose([rescale,
                                                            ToTensor()]))

train_loader = DataLoader(train_set, batch_size=50, shuffle=True)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 1
    total_step = len(train_loader)

    model = SigNet().to(device).float()

    # Make sure the first conv layer can process double tensors
    model.double()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train
    for epoch in range(num_epochs):
        for i_batch, sample_batched in enumerate(train_loader):
            # Forward Pass
            data = (sample_batched['img_a'].to(device), sample_batched['img_b'].to(device))
            outputs = model(data)

            loss = criterion(outputs, sample_batched['is_match'].reshape(-1))
            loss.backward()
            optimizer.step()

            # Backprop
            optimizer.zero_grad()

            if (i_batch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i_batch+1}/{total_step}], Loss: {loss.item():.4f}')


