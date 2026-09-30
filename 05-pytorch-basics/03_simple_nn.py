"""
Defining a feedforward neural network as an nn.Module subclass (Linear ->
ReLU -> Linear) and running a forward pass on a fake batch of input data.
"""

import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleNN(input_size=4, hidden_size=8, num_classes=3)
print(model)

# fake batch of data: 5 samples, 4 features each (like Iris)
sample_input = torch.randn(5, 4)
output = model(sample_input)
print("output shape:", output.shape)
print(output)
