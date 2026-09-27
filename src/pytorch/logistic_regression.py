import torch
from torch import nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 0. Data Preparation

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67
)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1. model
input_size = n_features


class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegression(n_input_features=input_size)

# 2. loss and optimizer

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. training loop
epochs = 200

for epoch in range(epochs):
    # forward pass
    y_pred = model(X_train)

    # loss function
    loss = criterion(y_pred, y_train)

    # back propagation
    loss.backward()

    # weight update
    optimizer.step()

    # clear gradients
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch + 1}, loss: {loss:.6f}")

# 4. Evaluation
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = y_pred.round()
    acc = y_pred_class.eq(y_test).sum() / float(y_test.shape[0])
    print(f"Accuracy: {acc:.4f}")
