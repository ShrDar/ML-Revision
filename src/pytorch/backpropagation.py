import torch

x = torch.tensor(10.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)

# forward pass and loss computation

y_pred = w * x + b
loss = (y_pred - y) ** 2

print(y_pred)

# backward pass

loss.backward()
print(w.grad)
