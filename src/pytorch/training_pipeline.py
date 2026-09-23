# 1.) Design Model (input size, output size, forward pass)
# 2.) Construct Loss and Optimizer
# 3.) Training Loop
#   - forward pass: prediction computation
#   - backward pass: gradients
#   - weight updation

import torch
from torch import nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearRegression(input_size, output_size)


print(f"Prediction before Training: f(5) = {model(X_test).item():.3f}")

# Training
learning_rate = 0.01
epochs = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    # prediction = forward_pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    l.backward()  # dJ/dw

    # update Weight
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()  # modifies in place

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"Epoch: {epoch + 1}, weight: {w[0][0].item():.3f}, loss = {l:.8f}")


print(f"Prediction after Training: f(5) = {model(X_test).item():.3f}")
