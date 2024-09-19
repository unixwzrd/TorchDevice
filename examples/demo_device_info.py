#!/usr/bin/env python

# Demo 5: Device Information

import torchdevice
import torch

def main():
    is_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    capability = torch.cuda.get_device_capability(current_device)

    print(f"CUDA Available: {is_available}")
    print(f"Device Count: {device_count}")
    print(f"Current Device Index: {current_device}")
    print(f"Device Name: {device_name}")
    print(f"Device Capability: {capability}")

if __name__ == '__main__':
    main()