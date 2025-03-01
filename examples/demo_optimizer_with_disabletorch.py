#!/usr/bin/env python3
# Demo: Neural Network Training with PyTorch Optimizer using DisableTorchFunction

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

def create_optimizer(model, optimizer_class=optim.SGD, **kwargs):
    """
    Create an optimizer safely by manually creating parameter groups
    """
    param_groups = []
    
    # Create parameter groups manually
    for param in model.parameters():
        if param.requires_grad:
            param_groups.append({'params': [param], **kwargs})
    
    # Create optimizer with the parameter groups
    if optimizer_class == optim.SGD:
        # For SGD, we can create it directly with our parameters
        return CustomSGD(param_groups, **kwargs)
    else:
        # For other optimizers, we might need different approaches
        return optimizer_class(model.parameters(), **kwargs)

class CustomSGD:
    """
    A simple implementation of SGD that doesn't rely on torch._dynamo
    """
    def __init__(self, param_groups, lr=0.01, momentum=0, dampening=0, 
                 weight_decay=0, nesterov=False):
        self.param_groups = param_groups
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # Initialize momentum buffers
        self.state = {}
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = {}
                if momentum > 0:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data)
    
    def zero_grad(self):
        """Zero out the gradients"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
    
    def step(self):
        """Perform a single optimization step"""
        for group in self.param_groups:
            weight_decay = group.get('weight_decay', self.weight_decay)
            momentum = group.get('momentum', self.momentum)
            dampening = group.get('dampening', self.dampening)
            nesterov = group.get('nesterov', self.nesterov)
            lr = group.get('lr', self.lr)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                # Apply weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    buf = self.state[p].get('momentum_buffer', None)
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        self.state[p]['momentum_buffer'] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                # Update parameters
                p.data.add_(d_p, alpha=-lr)

def main():
    # Set device (will use TorchDevice)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and move to device
    model = SimpleNN().to(device)
    
    # Create dummy data
    X = torch.randn(100, 10, device=device)
    y = torch.randn(100, 1, device=device)
    
    # Create our custom optimizer
    optimizer = create_optimizer(model, lr=0.01)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(5):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
