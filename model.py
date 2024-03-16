import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def predict_softmax(self, x: Tensor):
        return F.softmax(self(x), dim=1)

    def predict_argmax(self, x: Tensor):
        return self.predict_softmax(x).argmax(1)

    def predict_one_argmax(self, x: Tensor):
        output = self.predict_softmax(x)
        result = output.argmax(1).item()
        output = output.view(-1).tolist()
        return output, result


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: Tensor):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def predict_softmax(self, x: Tensor):
        return F.softmax(self(x), dim=1)

    def predict_argmax(self, x: Tensor):
        return self.predict_softmax(x).argmax(1)

    def predict_one_argmax(self, x: Tensor):
        output = self.predict_softmax(x)
        result = output.argmax(1).item()
        output = output.view(-1).tolist()
        return output, result
