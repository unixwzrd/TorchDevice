#!/usr/bin/env python

# Demo 1: Basic Tensor Computation

# TorchDevice needs to be imported first to patch torch functions
# even if not directly used in the code
import TorchDevice
import torch
import numpy as np


def main():
    # Select the default device
    device = torch.device('cuda')

    # Create NumPy arrays
    np_array1 = np.array([1, 2, 3], dtype=np.float32)
    np_array2 = np.array([4, 5, 6], dtype=np.float32)

    # Convert to PyTorch tensors and move to device
    tensor1 = torch.from_numpy(np_array1).to(device)
    tensor2 = torch.from_numpy(np_array2).to(device)

    # Perform tensor operations
    result = tensor1 + tensor2

    # Move result back to CPU and convert to NumPy
    result_np = result.cpu().numpy()

    print(f"Result: {result_np}")


if __name__ == '__main__':
    main()