#!/usr/bin/env python

# Example 5: Creating Tensors Directly on the Device

import sys
sys.path.insert(0, '../pylib')
import torchdevice
import torch

device = torch.device('cuda')

# Create a tensor directly on the selected device
tensor = torch.ones(5, device=device)
print(f"Tensor on {device.type}: {tensor}")