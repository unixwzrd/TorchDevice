#!/usr/bin/env python

# Example 2: Using the Device Context Manager

import torchdevice
import torch

with torch.cuda.device(0):
    # Tensor operations within the context manager
    tensor = torch.tensor([1.0, 2.0, 3.0]).to()
    print(f"Tensor device: {tensor.device}")

    # Perform computations
    result = tensor + 5
    print(f"Computation result: {result}")