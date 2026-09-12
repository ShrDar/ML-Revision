import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()

        self.fc1 = nn.Linear(input_size, 55)
        self.fc2 = nn.Linear(55, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)


input_size = 784
num_classes = 10
batch_size = 32
learning_rate = 0.001
