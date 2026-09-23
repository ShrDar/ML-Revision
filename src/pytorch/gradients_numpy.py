import numpy as np

# f = w * x

# f = 2 * x

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0


# model prediction
def forward(x):
    return w * x


# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()


# gradient
# MSE = 1/N (w * x - y) ** 2
# dJ/dw = 2/N (wx - y) x


def gradient(x, y, y_pred):
    return (2 * x * (y_pred - y)).mean()


print(f"Prediction before Training: f(5) = {forward(5):.3f}")

# Training
learning_rate = 0.01
epochs = 10

for epoch in range(epochs):
    # prediction = forward_pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)

    # update Weight

    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f"Epoch: {epoch + 1}, weight: {w:.3f}, loss = {l:.8f}")


print(f"Prediction After Training: f(5) => {forward(5):.3f}")
