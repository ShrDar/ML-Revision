import torch

batch_size = 10
features = 25

x = torch.rand((batch_size, features))

# Fancy Indexing
x = torch.arange(10)
indices = [2, 5, 8]

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x)
print(x[rows, cols])  # (1, 4) and (0, 0) elements from the matrix

# Advanced Indexing
x = torch.arange(11)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])  # instead of .remainder "%" can also be used

# Useful Operations
print(x)
print(
    torch.where(x > 5, x, x * 2)
)  # if x is greater than 5 don't do anything else x * 2

print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique())
x = torch.rand((4, 5))
print(x.ndimension())
print(x.numel())
