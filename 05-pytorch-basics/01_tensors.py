"""
Creating tensors (from lists, zeros/rand, and NumPy arrays), basic arithmetic,
reshaping with view()/reshape(), and checking for GPU availability.
"""

import torch
import numpy as np

# creating tensors
x = torch.tensor([1, 2, 3])
y = torch.zeros(2, 3)
z = torch.rand(2, 3)

print(x)
print(y)
print(z)

# from numpy
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)
print(t)

# basic ops
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)
print(a * b)
print(a.dot(b))

# reshaping
c = torch.arange(6)
print(c.view(2, 3))
print(c.reshape(3, 2))

# device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
