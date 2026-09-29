import numpy as np
import torch
from torch import nn


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(f"Softmax Numpy: {outputs}")

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print("Softmax PyTorch: ", outputs)


def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss


# y shall be one hot encoded
# if class 0: [1, 0, 0]
# if class 1: [0, 1, 0]
# if class 2: [0, 0, 1]

Y = np.array([1, 0, 0])

Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.2, 0.2, 0.1])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f"Loss 1 numpy: {l1:.4f}")
print(f"Loss 2 numpy: {l2:.4f}")


loss = nn.CrossEntropyLoss()

Y = torch.tensor([2, 0, 1])
# nsamples * nclasses = 3x3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [5.0, 0.9, 2.6], [0.4, 4.1, 2.1]])
Y_pred_bad = torch.tensor([[0.5, 1.0, 0.1], [0.5, 2.0, 0.1], [4.5, 1.0, 2.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f"Loss 1: {l1.item()}")
print(f"Loss 2: {l2.item()}")


_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)
