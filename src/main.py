import torch
from torch import nn

X = torch.tensor(
    [
        [1.0, 10.0],
        [2.0, 30.0],
        [3.0, 29.0],
        [4.0, 30.0],
    ],
    dtype=torch.float32,
)
Y = torch.tensor(
    [
        [3.0],
        [6.0],
        [9.0],
        [12.0],
    ],
    dtype=torch.float32,
)

X_test = torch.tensor([2.0, 30.0])

n_samples, n_features = X.shape
input_size = n_features
output_size = 1


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel(input_size, output_size)

print(f"Prediction before Training: {model(X_test)}")

learning_rate = 0.001
epochs = 200

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(epochs):
    y_pred = model(X)

    l = loss(y_pred, Y)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"Epoch: {epoch + 1}, weight: {w[0][0].item()}, loss: {l}")

print(f"Prediction After Training: {model(X_test)}")
