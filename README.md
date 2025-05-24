# TorchDevice

TorchDevice is a Python library that enables transparent code portability between NVIDIA CUDA and Apple Silicon (MPS) hardware for PyTorch applications. It intercepts PyTorch calls related to GPU hardware, allowing developers to write code that works seamlessly on both accelerator types without modification. TorchDevice is designed primarily to help port code (for example, HuggingFace Transformers) written for CUDA to run on Apple Silicon (MPS) and vice versa.

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
    - [Device Redirection Policy](#device-redirection-policy)
    - [Usage with Optimizers](#usage-with-optimizers)
  - [Demo Scripts](#demo-scripts)
  - [Limitations](#limitations)
  - [Recent Updates](#recent-updates)
    - [March 2025 Updates](#march-2025-updates)
      - [CPU Override Feature](#cpu-override-feature)
      - [Improved Logging System](#improved-logging-system)
      - [Stream and Event Handling](#stream-and-event-handling)
      - [PyTorch Compatibility](#pytorch-compatibility)
      - [Code Quality](#code-quality)
    - [Upcoming Changes](#upcoming-changes)
  - [Project Status and Next Steps](#project-status-and-next-steps)
  - [Contributing](#contributing)
  - [Supporting the Development of this Project](#supporting-the-development-of-this-project)
  - [License](#license)

## Features

- **Automatic Device Redirection**: Intercepts `torch.device` instantiation and related tensor creation calls, redirecting them to the default accelerator (MPS or CUDA) when available.
- **Explicit CPU Override**: Provides a special `'cpu:-1'` device specification to force CPU usage regardless of available accelerators.
- **Mocked CUDA Functions**: Mocks CUDA-specific functions so that CUDA code can run on MPS hardware.
- **Stream and Event Support**: Implements support for CUDA streams and events on MPS devices for asynchronous operations and event timing.
- **Unified Memory Handling**: Provides reasonable memory-related outputs, bridging differences between CUDA and MPS.
- **Logging and Debugging**: Outputs detailed log messages indicating the interception and redirection of Torch calls, helping to identify areas in the code that may need migration.
- **Transparent Integration**: Works seamlessly with existing codebases without modification.
- **PyTorch Compiler Compatibility**: Fully compatible with PyTorch's dynamo compiler and inductor.

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch installed with the appropriate support for your hardware:
  - CUDA support for NVIDIA GPUs
  - MPS support for Apple Silicon (macOS 12.3+)
- Additional packages:
  - `numpy`
  - `psutil`

### Installing PyTorch

Follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for your platform.

#### Nightly builds for Apple Silicon

For the latest builds on Apple Silicon, install via pip:

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
   Or manually:  
   ```bash
   pip install numpy psutil
   ```

#### NumPy and Apple Silicon

**IMPORTANT – For Apple Silicon, ensure NumPy is linked to the Accelerate Framework** to obtain improved performance for vector operations. (See [VenvUtil](https://github.com/unixwzrd/VenvUtil) for the latest command.)

4. **Install TorchDevice**  
   You can install it as follows:  
   ```bash
   python setup.py install
   ```  
   Or via pip:  
   ```bash
   pip install .
   ```

## Usage

Import `TorchDevice` **before** importing `torch` to automatically apply the patches.

```python
import TorchDevice  # Must be imported first.
import torch

device = torch.device('cuda')  # On unsupported hardware (e.g. Apple Silicon), this will be redirected (e.g., to "mps").
# Your existing PyTorch code runs unmodified.
```

### Important Notes

- **Device Selection**: TorchDevice selects an appropriate device based on hardware availability:
  - If CUDA is requested but unavailable, it redirects to MPS if available.
  - If MPS is requested but unavailable, it redirects to CUDA if available.
  - If no accelerator is available, it defaults to CPU.
- **Logging**: Log messages detail intercepted and redirected calls, including caller information.
- **Unsupported Functions**: Certain functions not supported on specific hardware are stubbed out, with warnings logged.
- **Stream and Event Support**: CUDA streams and events are fully supported on MPS devices.

### Device Redirection Policy

For a full list of device redirection policy, see the [TorchDevice Redirection Behavior](#torchdevice-redirection-behavior) section.

### Usage with Optimizers

TorchDevice works seamlessly with PyTorch optimizers without special configuration. For example:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from TorchDevice import TorchDevice

# Obtain the configured device
device = TorchDevice()
print(f"Using device: {device}")

# Create a model and move it to the device
model = nn.Linear(10, 2).to(device.device)

# Create an optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop operates normally
```

## Demo Scripts

Demo scripts are provided to showcase TorchDevice usage:
- `demo_basic_tensor.py`
- `demo_matrix_multiplication.py`
- `demo_neural_network.py`
- `demo_unsupported_functions.py`
- `demo_device_info.py`

Ensure these scripts are updated to account for any device redirection behavior.

## Limitations

- **Precision Support on MPS:** MPS does not support `float64` (double precision); use `float32` instead.
- **Multiple Devices on MPS:** MPS does not support multiple devices; device queries and settings may be redirected.
- **Partial CUDA Functionality:** While many CUDA functions are mocked, some features cannot be fully replicated on MPS hardware.
- **Performance Considerations:** Mocked functions may not accurately reflect hardware performance.
- **Tensor Creation:** Some direct tensor creation operations may be forcibly redirected to the accelerator, which could require code adjustments if genuine CPU tensors are needed.

## Recent Updates

### March 2025 Updates

#### CPU Override Feature

- Implemented the special `'cpu:-1'` device specification to force CPU usage.
- Added a CPU override flag ensuring that all explicit CPU requests produce CPU tensors until the override is toggled off.
- Enhanced device redirection logic for clarity and maintainability.

#### Improved Logging System

- Consolidated and enhanced logging via the TDLogger module.
- Standardized log messages including caller filename, function name, and line number.
- Enhanced log filtering and formatting for better diagnostics.

#### Stream and Event Handling

- Implemented comprehensive CUDA stream functionality on MPS.
- Added robust support for CUDA event handling, including proper timing and synchronization.

#### PyTorch Compatibility

- Ensured compatibility with PyTorch's dynamo compiler and inductor.
- Improved compatibility with PyTorch optimizers.
- Reduced patched functions to the essential minimum.

#### Code Quality

- Improved error handling and message logging for easier debugging.
- Increased test coverage for streams, events, and device conversion.
- Refactored code for better maintainability.

### Upcoming Changes

- **Modular Architecture:** Further breakdown of the codebase into smaller, focused modules.
- **Enhanced Testing:** Additional unit tests for individual components.
- **Better Documentation:** Expanded examples and detailed API references.
- **Configurable Logging:** More detailed and configurable log options.

## Project Status and Next Steps

- **Modularization Complete:** All core logic has been modularized into dedicated files (see `TorchDevice/cuda/`).
- **CPU Override Feature:** The `cpu:-1` syntax is fully supported and documented. See below and the API docs for usage.
- **All Core Tests Passing:** All unit and integration tests for the modularized codebase are passing as of the latest commit.
- **Pre-Commit Checklist:** See `.project-planning/pre-commit-checklist.md` for quality, testing, and documentation standards. All items except multi-version/hardware testing and post-commit tasks are complete.
- **Next Focus:** Running and fixing all example/demo scripts in the `examples/` directory, and expanding user-facing features.
- **Contributor Guidance:** For ongoing work, see `docs/modularization-plan.md` and `.project-planning/pre-commit-checklist.md` for up-to-date progress and standards.

## Contributing

Contributions are welcome! Please:

1. Report issues on GitHub.
2. Submit pull requests for bug fixes or new features.
3. Improve documentation by fixing errors or adding examples.
4. Test TorchDevice on various hardware configurations and report your findings.

Follow these guidelines:
- Conform to the existing code style.
- Add tests for new features.
- Update documentation accordingly.
- Ensure all tests pass prior to submission.

## Supporting the Development of this Project

For more information and other projects check out the [Distributed Thinking Systems](https://unixwzrd.ai) website.

If you find TorchDevice useful, please consider donating or sponsoring the project:

[unixwzrd Patreon](https://www.patreon.com/unixwzrd)  
[unixwzrd Ko-Fi](https://ko-fi.com/unixwzrd)  
[unixwzrd Buy Me a Coffee](https://www.buymeacoffee.com/unixwzrd)

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