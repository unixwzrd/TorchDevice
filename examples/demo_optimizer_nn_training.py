#!/usr/bin/env python

# Demo: Neural Network Training with PyTorch Optimizer

import os
import sys

# Add the parent directory to sys.path to find the TorchDevice package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import TorchDevice first before any torch imports
import TorchDevice  # noqa: F401

# Only after TorchDevice import, import torch and its modules
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    print("Creating device...")
    try:
        device = torch.device('cuda')
        print(f"Device created: {device}")
    except Exception as e:
        print(f"Warning: Error creating device: {e}")
        device = torch.device('cpu')
        print(f"Fallback device created: {device}")

    # Simple dataset - handle error cases gracefully
    print("Creating tensors...")
    try:
        x = torch.linspace(-1, 1, 100).unsqueeze(1).to(device)
        y = x.pow(3) + 0.3 * torch.rand(x.size()).to(device)
        print(f"Tensors created on {device}")
    except Exception as e:
        print(f"Warning: Error creating tensors on device {device}: {e}")
        x = torch.linspace(-1, 1, 100).unsqueeze(1)
        y = x.pow(3) + 0.3 * torch.rand(x.size())
        print("Tensors created on CPU")
        device = torch.device('cpu')

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
    
    # Create optimizer, using a try/except block to handle any issues
    try:
        # Try to create optimizer directly
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    except Exception as e:
        print(f"Warning: Error creating optimizer: {e}")
        # Fall back to CPU tensors if needed
        device = torch.device('cpu')
        model = model.to(device)
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
        
        # Optimizer step with error handling
        try:
            optimizer.step()
        except Exception as e:
            print(f"Warning: Error during optimizer.step(): {e}")
            # Try one more time
            try:
                optimizer.step()
            except Exception as inner_e:
                print(f"Error: Unable to run optimizer.step(): {inner_e}")
                # Skip this step
                continue

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
