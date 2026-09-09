import torch

x = torch.arange(9)

x_3x3 = x.view(3, 3)  # it need to be in a contiguious block of memory
x_3x3 = x.reshape(3, 3)  # it doesn't need to be in a contiguous block of memory

y = x_3x3.t()

print(
    y.view(9)
)  # doesn't work as the indexes are shifted which makes it non-contiguous


print(y.reshape(9))  # works as it doesn't need contiguious memory

print(
    y.contiguous().view(9)
)  # this works as it's first transformed into contiguious block


x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))

print(x1)
print(x2)

print(torch.cat((x1, x2), dim=0))
print(torch.cat((x1, x2), dim=1))

batch = 64
x = torch.rand((batch, 2, 5))

print(x.view(batch, -1))

z = x.permute((0, 2, 1))
print(z.shape)

x = torch.arange(10)
print(x.shape)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)

print(x.shape)
print(x)

z = x.squeeze(1)
print(z.shape)
