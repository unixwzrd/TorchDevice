#!/usr/bin/env python

# Demo 2: Matrix Multiplication

import TorchDevice
import torch
import numpy as np

def main():
    device = torch.device('cuda')

    # Create random matrices
    np_matrix1 = np.random.rand(3, 3).astype(np.float32)
    np_matrix2 = np.random.rand(3, 3).astype(np.float32)

    # Convert to tensors
    tensor_matrix1 = torch.from_numpy(np_matrix1).to(device)
    tensor_matrix2 = torch.from_numpy(np_matrix2).to(device)

    # Matrix multiplication
    result = torch.matmul(tensor_matrix1, tensor_matrix2)

    # Move result back to CPU
    result_cpu = result.cpu()

    print(f"Matrix 1:\n{np_matrix1}")
    print(f"Matrix 2:\n{np_matrix2}")
    print(f"Result:\n{result_cpu.numpy()}")

if __name__ == '__main__':
    main()