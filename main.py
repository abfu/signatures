"""
For now focus on wether two signatures are matches or not.

Afterwards add forgery detection to network.

"""
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from utils import SignatureDataset, Rescale, ToTensor
from model import SigNet
import argparse

DATADIR = './data/sign_data/'

rescale = Rescale((100, 100))

# TODO Speed up data loader & add cuda support
train_set = SignatureDataset(root='./data/sign_data/',
                             split='train',
                             transforms=transforms.Compose([rescale,
                                                            ToTensor()]))
test_set = SignatureDataset(root='./data/sign_Data/',
                            split='test',
                            transforms=transforms.Compose([rescale,
                                                           ToTensor()]))



if __name__ == '__main__':
    # TODO: Add argparser
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 150
    num_workers = 8  # Set number of workers for prefetching of data


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_epochs = 1
    total_step = len(train_loader)

    model = SigNet().to(device).float()

    # Make sure the first conv layer can process double tensors
    model.double()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    for epoch in range(num_epochs):
        for i_batch, sample_batched in enumerate(train_loader):
            # Forward Pass
            data = (sample_batched['img_a'].to(device), sample_batched['img_b'].to(device))
            outputs = model(datat)

            optimizer.zero_grad()
            loss = criterion(outputs, sample_batched['is_match'].reshape(-1))
            loss.backward()
            optimizer.step()


            if (i_batch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i_batch+1}/{total_step}], Loss: {loss.item():.4f}')

                if i_batch + 1 == 20:
                    break

    # Eval
    with torch.no_grad():
        correct = 0
        total = 0
        for i_batch, sample_batched in enumerate(test_loader):
            data = (sample_batched['img_a'].to(device), sample_batched['img_b'].to(device))
            labels = sample_batched['is_match'].to(device)
            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.reshape(-1)).sum().item()
        print(f'Test accuracy of the model on {total} test images: {100 * correct / total}')

    torch.save(model.state_dict(), 'model_sig.ckpt')
