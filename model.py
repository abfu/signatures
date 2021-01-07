import torch.nn as nn
import torch.nn.functional as F
import torch


class SigNet(nn.Module):
    def __init__(self):
        super(SigNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=18, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=52, kernel_size=3, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(26 * 26 * 52, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, data):
        res = []
        for img in data:
            out = self.conv1(img)
            out = F.relu(out)
            out = self.pool1(out)
            out = self.conv2(out)
            out = F.relu(out)
            out = self.pool2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            res.append(F.relu(out))

        res = torch.abs(res[1] - res[0])
        res = self.fc2(res)
        return res
