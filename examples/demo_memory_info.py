#!/usr/bin/env python

# Example 3: Memory Information

import sys
sys.path.insert(0, '../pylib')
import TorchDevice
import torch

allocated = torch.cuda.memory_allocated()
reserved = torch.cuda.memory_reserved()

print(f"Memory Allocated: {allocated} bytes")
print(f"Memory Reserved: {reserved} bytes")