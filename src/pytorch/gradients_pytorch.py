import torch

# f = w * x

# f = 2 * x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# model prediction
def forward(x):
    return w * x


# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()


print(f"Prediction before Training: f(5) = {forward(5):.3f}")

# Training
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    # prediction = forward_pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    l.backward()  # dJ/dw

    dw = w.grad

    # update Weight
    with torch.no_grad():
        w -= learning_rate * dw

    # zero gradients
    w.grad.zero_()  # modifies in place

    if epoch % 10 == 0:
        print(f"Epoch: {epoch + 1}, weight: {w:.3f}, loss = {l:.8f}")


print(f"Prediction After Training: f(5) => {forward(5):.3f}")
