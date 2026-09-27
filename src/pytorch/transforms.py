import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset


class WineDataset(Dataset):
    def __init__(self, transform=None):
        rawdata = np.loadtxt(
            "../../data/wine.csv",
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
        )
        self.x = rawdata[:, 1:]
        self.y = rawdata[:, 0:1]
        self.n_samples = self.x.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features)

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features)
