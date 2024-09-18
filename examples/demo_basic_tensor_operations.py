#!/usr/bin/env python

# Example 1: Basic Tensor Operations

import TorchDevice
import torch
import numpy as np

# Select the default device
device = torch.device('cuda')  # Will be redirected if necessary

# Create a NumPy array and convert it to a PyTorch tensor
np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
tensor = torch.from_numpy(np_array).to(device)

# Perform a simple operation
result = tensor * 2

print(f"Result on {device.type}: {result}")