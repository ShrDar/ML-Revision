import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
z2 = torch.add(x, y)
z = x + y

# Subtraction
z = x - y

# Division
z = torch.true_divide(x, y)  # performs element wise division if they're of equal shape

# Inplace operations
t = torch.zeros(3)
t.add_(x)  # _ means inplace
t += x  # t += x is not equal to t = t + x it creates a copy first
t = t + x

# Exponentiation
z = x.pow(2)
z = x**2

# Simple Comparision
z = x > 1

# Matrix Multiplication

x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)
print(x3)

# Matrix Exponentitation
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3)  # Matrix * Matrix * Matrix

# Element Wise Multiplication
z = x * y

# dot product
z = torch.dot(x, y)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)

# Example of broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))
z = x1 - x2
z = x1**x2

# Other useful tensor operations
print(x)
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y)
z = torch.not_equal(x, y)
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(
    x, min=0, max=10
)  # if any value less than 0 change it to 10 or clamp it to 10 and if any value greater than 10 then clamp it to 10

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x)  # only one value of x needs to be true for it to be true
z = torch.all(x)  # all of the value needs to be true to be true

print(z)
