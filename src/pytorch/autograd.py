import torch

x = torch.tensor(4.0)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = 77

epochs = 30
lr = 0.01

for epoch in range(epochs):
    y_pred = w * x + b

    loss = (1 / 2) * (y_pred - y) ** 2

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    print(f"Epoch: {epoch}, Loss: {loss.item()}, Predicted: {y_pred.item()}")
