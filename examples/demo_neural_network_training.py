#!/usr/bin/env python3
# Demo: Neural Network Training with PyTorch Optimizer

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Disable torch compile to avoid issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_INDUCTOR_DISABLE"] = "1"

# Import TorchDevice after setting environment variables
from TorchDevice import TorchDevice

# Simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def manual_sgd_step(model, lr=0.01):
    """
    Manually update model parameters using SGD
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data.sub_(lr * param.grad)

def main():
    # Set device (will use TorchDevice)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and move to device
    model = SimpleNN().to(device)
    
    # Create dummy data
    X = torch.randn(100, 10, device=device)
    y = torch.randn(100, 1, device=device)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(5):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Manual SGD step instead of using optimizer
        manual_sgd_step(model, lr=0.01)
        
        # Zero gradients for next iteration
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()