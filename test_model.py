from model import SigNet
import torch.nn as nn
import torch


def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SigNet().to(device)
    model.double()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    img_a = torch.rand(1, 3, 100, 100).double().to(device)
    img_b = torch.rand(1, 3, 100, 100).double().to(device)
    target = torch.ones(1, dtype=torch.long)
    data = (img_a, img_b)
    outputs = model(data)
    optimizer.zero_grad()
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()
