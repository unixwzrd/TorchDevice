#!/usr/bin/env python

# Demo: Neural Network with Manual Training Loop (no optimizer)

import os
# Disable PyTorch compiler (torch._dynamo) which is causing issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import TorchDevice
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def main():
    print("\n=== Testing TorchDevice with Manual Neural Network Training ===\n")
    
    # Create a device using CUDA syntax (will be redirected if on MPS)
    print("Creating device...")
    device = torch.device('cuda')
    print(f"Device created: {device}")
    
    # Create a simple dataset
    print("\nCreating dataset...")
    x = torch.linspace(-1, 1, 100).unsqueeze(1).to(device)
    y = x.pow(3) + 0.3 * torch.rand(x.size()).to(device)
    print(f"Dataset created on {device}")
    
    # Create a simple neural network
    print("\nCreating neural network...")
    model = nn.Sequential(
        nn.Linear(1, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)
    print(f"Model created on {device}")
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Manual training loop (no optimizer)
    print("\nStarting manual training loop...")
    losses = []
    learning_rate = 0.01
    for epoch in range(100):
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        losses.append(loss.item())
        
        # Manual backward pass
        model.zero_grad()
        loss.backward()
        
        # Manual parameter update (instead of optimizer.step())
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")
    
    # Test the model
    print("\nTesting model...")
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor([[0.5]]).to(device)
        prediction = model(test_input)
    print(f"Prediction for input 0.5: {prediction.item():.4f}")
    
    # Plot the loss curve
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_loss.png')
        print("\nLoss curve saved as 'training_loss.png'")
    except Exception as e:
        print(f"\nCould not save loss curve: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == '__main__':
    main()
