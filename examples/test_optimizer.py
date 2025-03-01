#!/usr/bin/env python3
"""
Simple test script to verify that the TorchDevice works correctly with optimizers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from TorchDevice import TorchDevice

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Create a model
    model = SimpleModel()
    
    # Get the available device
    device = TorchDevice()
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device.device)
    
    # Create a simple optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create dummy data
    inputs = torch.randn(4, 10).to(device.device)
    targets = torch.randn(4, 2).to(device.device)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(5):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
