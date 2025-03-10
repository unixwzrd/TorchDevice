# TorchDevice

TorchDevice is a Python library that enables transparent code portability between NVIDIA CUDA and Apple Silicon (MPS) hardware for PyTorch applications. It intercepts PyTorch calls related to GPU hardware, allowing developers to write code that works seamlessly on both hardware types without modification. TorchDevice is designed to assist in porting code from CUDA to MPS and vice versa, making it easier to develop cross-platform PyTorch applications.

The primary goal of this project for now is to be able to get HuggingFace Transformers working on Apple Silicon.

## Table of Contents

- [TorchDevice](#torchdevice)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Installing PyTorch](#installing-pytorch)
      - [Nightly builds for Apple Silicon](#nightly-builds-for-apple-silicon)
    - [Installing TorchDevice](#installing-torchdevice)
      - [NumPy and Apple Silicon](#numpy-and-apple-silicon)
  - [Usage](#usage)
    - [Important Notes](#important-notes)
    - [CPU Override Feature](#cpu-override-feature)
  - [Usage with Optimizers](#usage-with-optimizers)
  - [Demo Scripts](#demo-scripts)
  - [Limitations](#limitations)
  - [Recent Updates](#recent-updates)
    - [March 2025 Updates](#march-2025-updates)
      - [CPU Override Feature](#cpu-override-feature-1)
      - [Improved Logging System](#improved-logging-system)
      - [Stream and Event Handling](#stream-and-event-handling)
      - [PyTorch Compatibility](#pytorch-compatibility)
      - [Code Quality](#code-quality)
    - [Upcoming Changes](#upcoming-changes)
  - [Contributing](#contributing)
  - [Supporting the development of this project](#supporting-the-development-of-this-project)
  - [License](#license)

## Features

- **Automatic Device Redirection**: Intercepts `torch.device` instantiation and redirects it based on available hardware (CUDA, MPS, or CPU).
- **Explicit CPU Override**: Provides a special `'cpu:-1'` device specification to force CPU usage regardless of available accelerators.
- **Mocked CUDA Functions**: Provides mocked implementations of CUDA-specific functions, enabling code that uses CUDA functions to run on MPS hardware.
- **Stream and Event Support**: Implements full support for CUDA streams and events on MPS devices, allowing for asynchronous operations and event timing.
- **Unified Memory Handling**: Handles differences in memory management between CUDA and MPS, providing reasonable values for memory-related functions.
- **Logging and Debugging**: Outputs informative log messages indicating how calls are intercepted and handled, assisting in code migration and debugging.
- **Transparent Integration**: Works transparently without requiring changes to existing codebases.
- **PyTorch Compiler Compatibility**: Works seamlessly with PyTorch's dynamo compiler and inductor.

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch installed with appropriate support for your hardware:
  - For CUDA support on NVIDIA GPUs
  - For MPS support on Apple Silicon (macOS 12.3+)
- Additional Python packages:
  - `numpy`
  - `psutil`

### Installing PyTorch

Follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) to install PyTorch with the appropriate support for your hardware.

#### Nightly builds for Apple Silicon

You may want the latest nightly builds for Apple Silicon, use pip to install them:

```bash
pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```

### Installing TorchDevice

1. **Clone the Repository**

   ```bash
   git clone https://github.com/unixwzrd/TorchDevice.git
   ```

2. **Navigate to the Project Directory**

   ```bash
   cd TorchDevice
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *Alternatively, install dependencies manually:*

   ```bash
   pip install numpy psutil
   ```

#### NumPy and Apple Silicon

   **IMPORTANT - For Apple Silicon, you will want a NumPY linked to the accelerate framework.**
   This should always be done in the even NumPy gets downgraded or overlaid by another package, (SciPy), etc. The binaries as far as I can tell are not linked to the Apple Accelerate Framework, and NumPy does a lot of heavy lifting for PyTorch. Doing this can result in about an 8x performance improvement for vector operations.

   Here is the way you can ensure NumPy is linked properly for your machine;

   **NOTE: This command line may not be up to date, check the [VenvUtil](https://github.com/unixwzrd/VenvUtil) project for the latest version.**

   ```bash
    # NumPy Rebuild with Pip
    PATH="/usr/bin:${PATH}" CFLAGS="-I/System/Library/Frameworks/vecLib.framework/Headers -fno-strict-aliasing -DHAVE_BLAS_ILP64 -DACCELERATE_NEW_LAPACK=1 -DACCELERATE_LAPACK_ILP64=1" pip install numpy=="$VERSION" --force-reinstall --no-deps --no-cache --no-binary :all: --no-build-isolation --compile -Csetup-args=-Dblas=accelerate -Csetup-args=-Dlapack=accelerate -Csetup-args=-Duse-ilp64=true
   ```

There are additional tools for handling Python Virtual Environments as well as recompiling NumPy to ensure it is linked to the Accelerate Framework.

[VenvUtil - Virtual Environment Utility](https://github.com/unixwzrd/VenvUtil)

4. **Install TorchDevice Module**

   Since `TorchDevice` is a single Python file, you can copy `TorchDevice.py` to your project's directory or install it as a package:

   ```bash
   python setup.py install
   ```

   *Alternatively, install `TorchDevice` as a package this way as well:*

   ```bash
   pip install .
   ```

## Usage

Import `TorchDevice` in your code before importing `torch`. The module will automatically apply patches to intercept and redirect PyTorch calls.

```python
import TorchDevice  # Import in any order.
import torch

device = torch.device('cuda')  # This will be redirected based on available hardware
# For example, on Apple Silicon without CUDA, this will be redirected to MPS

# Your existing PyTorch code works without modification
```

### Important Notes

- **Device Selection**: `TorchDevice` will select the appropriate device based on hardware availability:
  - If CUDA is requested but not available, it will redirect to MPS if available.
  - If MPS is requested but not available, it will redirect to CUDA if available.
  - If neither is available, it will default to CPU.
- **Logging**: The module outputs log messages indicating how calls are intercepted and handled. These messages include the caller's filename, function name, and line number.
- **Log File**: You can direct logs to a file by setting the `TORCHDEVICE_LOG_FILE` environment variable.
- **Unsupported Functions**: Functions that are not supported on the current hardware are stubbed and will log a warning but allow execution to continue.
- **Stream and Event Support**: CUDA streams and events are fully supported on MPS devices, allowing for asynchronous operations and event timing.

### CPU Override Feature

TorchDevice now supports explicitly forcing CPU usage regardless of available accelerators using the special `'cpu:-1'` device specification:

```python
import TorchDevice
import torch

# Force CPU usage regardless of available GPUs
device = torch.device('cpu:-1')

# All subsequent operations will use CPU
tensor = torch.randn(5, 5)  # Will be created on CPU
model = torch.nn.Linear(10, 5).to(device)  # Will be moved to CPU

# Even explicit GPU requests will be redirected to CPU
gpu_tensor = torch.randn(5, 5, device='cuda')  # Will still use CPU
```

This feature is useful for:
- Debugging GPU code on CPU
- Running specific operations on CPU while keeping others on GPU
- Ensuring consistent behavior across different hardware environments
- Benchmarking CPU vs GPU performance

Once the CPU override is activated with `'cpu:-1'`, it will remain active for the duration of the Python process. All subsequent device creations will respect this override.

## Usage with Optimizers

TorchDevice now works seamlessly with PyTorch optimizers without any special configuration. Here's a simple example:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from TorchDevice import TorchDevice

# Get the available device
device = TorchDevice()
print(f"Using device: {device}")

# Create a model and move it to the device
model = nn.Linear(10, 2).to(device.device)

# Create a simple optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop works as expected
```

## Demo Scripts

You can include the demo scripts provided earlier in your project to showcase how to use `TorchDevice`.

- `demo_basic_tensor.py`
- `demo_matrix_multiplication.py`
- `demo_neural_network.py`
- `demo_unsupported_functions.py`
- `demo_device_info.py`

Ensure these scripts are updated with any changes you made to `TorchDevice.py`.

## Limitations

- **Precision Support on MPS**: The MPS backend does not support `float64` (double precision). Use `float32` instead.
- **Multiple Devices on MPS**: MPS does not support multiple devices. Calls to set or get devices will be redirected appropriately.
- **Partial CUDA Functionality**: While many CUDA functions are mocked, some functionality cannot be fully replicated on MPS hardware.
- **Performance Considerations**: Mocked functions may not reflect actual hardware performance or capabilities.
- **Tensor Creation**: Direct tensor creation operations using CUDA may still fail on MPS as not all CUDA operations have direct MPS equivalents. This is a limitation of the underlying PyTorch implementation rather than TorchDevice.

## Recent Updates

### March 2025 Updates

#### CPU Override Feature

- **Added Explicit CPU Device Selection**:
  - Implemented special `'cpu:-1'` device specification to force CPU usage regardless of available accelerators
  - Added CPU override flag to ensure all subsequent operations respect explicit CPU selection
  - Enhanced device redirection logic to recognize and honor CPU override requests
  - Simplified device handling for better maintainability and reliability
  - Improved testing infrastructure with dedicated test modules for CPU and MPS operations

#### Improved Logging System

- **TDLogger Module Enhancements**:
  - Consolidated duplicate code with dedicated helper functions for message filtering and formatting
  - Optimized message filtering logic with declarative pattern matching
  - Improved error handling with robust exception capture
  - Enhanced memory management using fixed-size collections for logging history
  - Centralized important message patterns for consistent filtering
  
- **Test Framework Improvements**:
  - Created standardized test infrastructure in the `common` directory
  - Implemented `PrefixedTestCase` class for consistent test behavior and logging
  - Added dedicated utilities for log capture and verification
  - Improved log message formatting and separation between test output and redirected messages
  - Enhanced test organization and discoverability

#### Stream and Event Handling

- **Enhanced Stream Support**: Implemented comprehensive CUDA stream functionality on MPS devices, including:
  - Basic stream operations (query, synchronize)
  - Context manager support with `__enter__` and `__exit__` methods
  - Stream event handling capabilities
  - Wait event and wait stream functionality
- **Improved Event Handling**: Fixed CUDA events handling to ensure proper redirection to MPS:
  - Added robust implementation for the `elapsed_time` method on MPS events
  - Improved the record method to properly handle stream parameters
  - Fixed synchronization and timing issues

#### PyTorch Compatibility

- **PyTorch Dynamo Support**: Ensured compatibility with PyTorch's dynamo compiler by implementing proper inheritance from base classes
- **Optimizer Compatibility**: Fixed compatibility issues with PyTorch optimizers
- **Reduced Patching**: Minimized the number of patched functions to the essential minimum

#### Code Quality

- **Proper Inheritance**: Implemented proper inheritance for Stream and Event classes from PyTorch's base classes
- **Improved Error Handling**: Added better error handling and logging for debugging
- **Test Coverage**: Added comprehensive tests for stream and event functionality

### Upcoming Changes

We're planning a major refactoring of the TorchDevice codebase to improve maintainability and extensibility:

- **Modular Architecture**: Breaking down the large single file into smaller, focused modules
- **Improved Testing**: Adding more unit tests for individual components
- **Better Documentation**: Enhancing documentation with examples and API references
- **Enhanced Logging**: Implementing more detailed and configurable logging

## Contributing

Contributions are welcome! Here's how you can contribute:

1. **Report Issues**: If you encounter any bugs or have feature requests, please open an issue.
2. **Submit Pull Requests**: Feel free to submit pull requests for bug fixes or new features.
3. **Improve Documentation**: Help improve the documentation by fixing errors or adding examples.
4. **Test on Different Hardware**: Test TorchDevice on different hardware configurations and report your findings.

Please follow these guidelines when contributing:

- Follow the existing code style
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting a pull request

## Supporting the development of this project

If you find this useful please consider donating or sponsoring this project to help support continued development. You may do so at the following link:

[unizwzrd Patreon](https://www.patreon.com/unizwzrd)
[unixwzrd Ko-Fi](https://ko-fi.com/unixwzrd)
[unixwzrd Buy Me a Coffee](https://www.buymeacoffee.com/unixwzrd)

Your support is greatly appreciated!

## License

```text
  This project is licensed under the Apache License
                  Version 2.0, License.

  Copyright 2025 Michael P. Sullivan - unixwzrd@unixwzrd.ai

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
```
