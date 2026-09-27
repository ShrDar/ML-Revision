"""

epochs = 1 forward and backward pass for all training samples

batch_size = number of training samples in one forward and backward pass

number of iterations = number of passes, each pass using [batch_size] number of samples

eg: 100 samples, batch_size = 20 => 100/20 = 5 iterations for 1 epoch

"""

import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt(
            "../../data/wine.csv",
            delimiter=",",
            dtype=np.float32,
            skiprows=1,
        )
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# trianing loop

epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)
print(total_samples, n_iterations)

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward
        if (i + 1) % 5 == 0:
            print(
                f"Epoch: {epoch + 1}/{epochs}, step {i + 1}/{n_iterations}, inputs {inputs.shape}"
            )
