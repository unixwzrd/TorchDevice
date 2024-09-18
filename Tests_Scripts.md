# TorchDevice Demo Scripts

## Test Scripts (Demos)

Below are several test scripts using NumPy arrays and PyTorch tensors that demonstrate how to use `TorchDevice`. You can include these as demos in your project.

### Demo 1: Basic Tensor Computation

```python
# demo_basic_tensor.py

import TorchDevice
import torch
import numpy as np

def main():
    # Select the default device
    device = torch.device('cuda')

    # Create NumPy arrays
    np_array1 = np.array([1, 2, 3], dtype=np.float32)
    np_array2 = np.array([4, 5, 6], dtype=np.float32)

    # Convert to PyTorch tensors and move to device
    tensor1 = torch.from_numpy(np_array1).to(device)
    tensor2 = torch.from_numpy(np_array2).to(device)

    # Perform tensor operations
    result = tensor1 + tensor2

    # Move result back to CPU and convert to NumPy
    result_np = result.cpu().numpy()

    print(f"Result: {result_np}")

if __name__ == '__main__':
    main()
```

**Usage:**

```bash
python demo_basic_tensor.py
```

---

### Demo 2: Matrix Multiplication

```python
# demo_matrix_multiplication.py

import TorchDevice
import torch
import numpy as np

def main():
    device = torch.device('cuda')

    # Create random matrices
    np_matrix1 = np.random.rand(3, 3).astype(np.float32)
    np_matrix2 = np.random.rand(3, 3).astype(np.float32)

    # Convert to tensors
    tensor_matrix1 = torch.from_numpy(np_matrix1).to(device)
    tensor_matrix2 = torch.from_numpy(np_matrix2).to(device)

    # Matrix multiplication
    result = torch.matmul(tensor_matrix1, tensor_matrix2)

    # Move result back to CPU
    result_cpu = result.cpu()

    print(f"Matrix 1:\n{np_matrix1}")
    print(f"Matrix 2:\n{np_matrix2}")
    print(f"Result:\n{result_cpu.numpy()}")

if __name__ == '__main__':
    main()
```

**Usage:**

```bash
python demo_matrix_multiplication.py
```

---

### Demo 3: Neural Network Training Loop

```python
# demo_neural_network.py

import TorchDevice
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
```

**Usage:**

```bash
python demo_neural_network.py
```

---

### Demo 4: Handling Unsupported Functions

```python
# demo_unsupported_functions.py

import TorchDevice
import torch

def main():
    device = torch.device('cuda')

    # Try to use an unsupported CUDA function
    torch.cuda.ipc_collect()

    # Proceed with other operations
    tensor = torch.tensor([1, 2, 3], device=device)
    print(f"Tensor on {device.type}: {tensor}")

if __name__ == '__main__':
    main()
```

**Usage:**

```bash
python demo_unsupported_functions.py
```

**Note:**

- This demo shows how the module handles unsupported functions by logging a warning but allowing the code to continue execution.

---

### Demo 5: Device Information

```python
# demo_device_info.py

import TorchDevice
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
```

**Usage:**

```bash
python demo_device_info.py
```