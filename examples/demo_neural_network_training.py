#!/usr/bin/env python

# Demo 3: Neural Network Training Loop

import torchdevice
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    device = torch.device('cuda')

    # Simple dataset
    x = torch.linspace(-1, 1, 100).unsqueeze(1).to(device)
    y = x.pow(3) + 0.3 * torch.rand(x.size()).to(device)

    # Define a simple neural network
    model = nn.Sequential(
        nn.Linear(1, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/200], Loss: {loss.item():.4f}')

    # Test the model
    model.eval()
    test_input = torch.tensor([[0.5]]).to(device)
    prediction = model(test_input)
    print(f'Prediction for input 0.5: {prediction.item():.4f}')

if __name__ == '__main__':
    main()