#!/usr/bin/env python

# Demo: Neural Network Training with PyTorch Optimizer

import os
# Disable PyTorch compiler (torch._dynamo) which is causing issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# Disable PyTorch inductor
os.environ["TORCH_INDUCTOR_DISABLE"] = "1"

# Import torch first
import torch

# Import the disable_torch_compile module to patch torch._dynamo and torch.compile
import sys
import os

# Add the parent directory to sys.path to find the TorchDevice package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TorchDevice.disable_torch_compile import apply_patches

# Apply patches to disable torch._dynamo and torch.compile
apply_patches()

# Now import TorchDevice after we've disabled _dynamo
import TorchDevice
import torch.nn as nn
import torch.optim as optim

def main():
    print("Creating device...")
    device = torch.device('cuda')
    print(f"Device created: {device}")

    # Simple dataset
    print("Creating tensors...")
    x = torch.linspace(-1, 1, 100).unsqueeze(1).to(device)
    y = x.pow(3) + 0.3 * torch.rand(x.size()).to(device)
    print(f"Tensors created on {device}")

    # Define a simple neural network
    print("Creating model...")
    model = nn.Sequential(
        nn.Linear(1, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)
    print(f"Model created on {device}")

    # Create optimizer and loss function
    print("Setting up optimizer...")
    criterion = nn.MSELoss()
    
    # Create optimizer with torch._C.DisableTorchFunction to avoid issues
    with torch._C.DisableTorchFunction():
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    print("Optimizer created")

    # Training loop
    print("Starting training...")
    for epoch in range(50):  # Reduced number of epochs for faster testing
        # Forward pass
        model.train()
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Use torch._C.DisableTorchFunction for optimizer.step() to avoid issues
        with torch._C.DisableTorchFunction():
            optimizer.step()

        if (epoch + 1) % 10 == 0:  # Print more frequently
            print(f'Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}')

    # Test the model
    print("Testing model...")
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor([[0.5]]).to(device)
        prediction = model(test_input)
    print(f"Prediction for input 0.5: {prediction.item():.4f}")
    print("Test complete!")

if __name__ == '__main__':
    main()
