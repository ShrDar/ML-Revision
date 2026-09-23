import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
Y = np.array([3, 6, 9, 12, 15], dtype=np.float32)

w = 0.0


def forward_prop(x, w):
    y = w * X
    return y


def loss_func(y_pred, y):
    loss = ((y_pred - y) ** 2).mean()
    return loss


def gradient(x, y_pred, y):
    grad = (2 / len(X)) * ((y_pred - y) * x).sum()
    return grad


epochs = 20
learning_rate = 0.01

for epoch in range(epochs):
    y_pred = forward_prop(X, w)

    loss = loss_func(y_pred, Y)

    dw = gradient(X, y_pred, Y)

    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f"Epoch: {epoch}, weight: {w}, loss: {loss}")


print(w * X)
