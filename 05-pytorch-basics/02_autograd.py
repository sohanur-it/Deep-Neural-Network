"""
How PyTorch's autograd works: tracking operations on tensors with
requires_grad=True, computing gradients with .backward(), and temporarily
disabling tracking with torch.no_grad().
"""

import torch

# requires_grad tracks operations for backprop
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward()
print("dy/dx at x=2:", x.grad)

# with vectors
w = torch.randn(3, requires_grad=True)
b = torch.randn(1, requires_grad=True)
data = torch.tensor([1.0, 2.0, 3.0])

out = (w * data).sum() + b
out.backward()

print("w.grad:", w.grad)
print("b.grad:", b.grad)

# stopping gradient tracking
with torch.no_grad():
    z = w * 2
    print("z requires_grad:", z.requires_grad)
