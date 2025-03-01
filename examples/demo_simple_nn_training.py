#!/usr/bin/env python

# Demo: Simple Neural Network Training Loop without using PyTorch optimizer

import os
# Disable PyTorch compiler (torch._dynamo) which is causing issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# Disable PyTorch inductor
os.environ["TORCH_INDUCTOR_DISABLE"] = "1"

# Import TorchDevice first to ensure it patches everything
import TorchDevice
import torch
import torch.nn as nn

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

    # Create loss function
    print("Setting up training...")
    criterion = nn.MSELoss()
    learning_rate = 0.01
    print("Training setup complete")

    # Custom training loop with manual parameter updates (no optimizer)
    print("Starting training...")
    for epoch in range(50):  # Reduced number of epochs for faster testing
        # Forward pass
        model.train()
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Manual parameter update (equivalent to SGD optimizer)
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

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
