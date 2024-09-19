#!/usr/bin/env python

# Demo 4: Handling Unsupported Functions

import sys
sys.path.insert(0, '../pylib')
import torchdevice
import torch

def main():
    device = torch.device('cuda')

    # Try to use an unsupported CUDA function
    torch.cuda.ipc_collect()

    # Proceed with other operations
    tensor = torch.tensor([1, 2, 3], device=device)
    print(f"Tensor on {device.type}: {tensor}")

if __name__ == '__main__':
    main()