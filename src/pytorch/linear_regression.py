import torch
from torch import nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# 0. Data preparation
X_numpy, y_numpy = datasets.make_regression(
    n_samples=100,
    n_features=1,
    noise=20,
    random_state=1,
)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)


n_samples, n_features = X.shape

# 1. model design
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

# 2. loss and optimizer
learning_rate = 0.01

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3. training loop
epochs = 150
for epoch in range(epochs):
    # forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # backward pass
    loss.backward()

    # weight update
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10 == 0:
        w, b = model.parameters()
        print(f"Epoch: {epoch}, weight: {w[0][0].item()}, loss: {loss}")

# 4. Visualization
predicted = model(X).detach().numpy()  # detach removes it from the computational graph
plt.plot(X_numpy, y_numpy, "ro")
plt.plot(X_numpy, predicted, "b")
plt.show()
