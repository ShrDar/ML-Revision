import numpy as np
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

# 0. Data Preparation
iris_dataset = datasets.load_iris()

X, y = iris_dataset["data"], iris_dataset["target"]

n_samples, n_features = X.shape
n_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.int64))
y_test = torch.from_numpy(y_test.astype(np.int64))


# 1. Model Creation


class LogisticModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
        )

    def forward(self, x):
        return self.linear(x)


model = LogisticModel(in_features=n_features, out_features=n_classes)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()


learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3. Training Loop
epochs = 200

for epoch in range(epochs):
    y_pred = model(X_train)

    loss = criterion(y_pred, y_train)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.5f}")

with torch.no_grad():
    y_test_pred = model(X_test)

    y_test_pred_class = torch.argmax(y_test_pred, dim=1)

    accuracy = (y_test_pred_class == y_test).float().mean()

    test_loss = criterion(y_test_pred, y_test)

    print(f"Test Loss: {test_loss.item():.4f}")
    print(f"Test Accuracy: {accuracy.item():.4f}")
