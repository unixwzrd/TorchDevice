# TorchDevice *WIP*`

TorchDevice is a class in the TorchDevice.py package that intercepts PyTorch calls related to GPU hardware, enabling transparent code portability between NVIDIA CUDA and Apple Silicon (MPS) hardware. It allows developers to write code that works seamlessly on both hardware types without modification and is meant to assist in porting code from CUDA to MPS.

## Table of Contents

- [TorchDevice *WIP*\`](#torchdevice-wip)
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
  - [Usage with Optimizers](#usage-with-optimizers)
  - [Demo Scripts](#demo-scripts)
  - [Limitations](#limitations)
  - [Recent Updates](#recent-updates)
  - [Contributing](#contributing)
  - [Consider supporting the development of this project](#consider-supporting-the-development-of-this-project)
  - [License](#license)

## Features

- **Automatic Device Redirection**: Intercepts `torch.device` instantiation and redirects it based on available hardware (CUDA, MPS, or CPU).
- **Mocked CUDA Functions**: Provides mocked implementations of CUDA-specific functions, enabling code that uses CUDA functions to run on MPS hardware.
- **Unified Memory Handling**: Handles differences in memory management between CUDA and MPS, providing reasonable values for memory-related functions.
- **Logging and Debugging**: Outputs informative log messages indicating how calls are intercepted and handled, assisting in code migration and debugging.
- **Transparent Integration**: Works transparently without requiring changes to existing codebases.

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

Import `TorchDevice` in your code before or after importing `torch`. The module will automatically apply patches to intercept and redirect PyTorch calls.

```python
import TorchDevice  # import TorchDevice to apply patches
import torch

device = torch.device('cuda')  # This will be redirected based on available hardware

# Your existing PyTorch code
```

### Important Notes

- **Device Selection**: `TorchDevice` will select the appropriate device based on hardware availability:
  - If CUDA is requested but not available, it will redirect to MPS if available.
  - If MPS is requested but not available, it will redirect to CUDA if available.
  - If neither is available, it will default to CPU.
- **Logging**: The module outputs log messages indicating how calls are intercepted and handled. These messages include the caller's filename, function name, and line number.
- **Unsupported Functions**: Functions that are not supported on the current hardware are stubbed and will log a warning but allow execution to continue.

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
# ...

## Demo Scripts

You can include the demo scripts provided earlier in your project to showcase how to use `TorchDevice`.

- `demo_basic_tensor.py`
- `demo_matrix_multiplication.py`
- `demo_neural_network.py`
- `demo_unsupported_functions.py`
- `demo_device_info.py`

Ensure these scripts are updated with any changes you've made to `TorchDevice.py`.

## Limitations

- **Precision Support on MPS**: The MPS backend does not support `float64` (double precision). Use `float32` instead.
- **Multiple Devices on MPS**: MPS does not support multiple devices. Calls to set or get devices will be redirected appropriately.
- **Partial CUDA Functionality**: While many CUDA functions are mocked, some functionality cannot be fully replicated on MPS hardware.
- **Performance Considerations**: Mocked functions may not reflect actual hardware performance or capabilities.

## Recent Updates

The TorchDevice implementation has been simplified to ensure seamless compatibility between CUDA and MPS devices without requiring any special handling for optimizers or disabling the PyTorch compiler.

Key improvements:
- Removed the need to disable PyTorch compiler and inductor
- Simplified Event and Stream class implementations
- Reduced the number of patched functions to the essential minimum
- Fixed compatibility issues with PyTorch optimizers

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss potential changes or improvements.

## Consider supporting the development of this project

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
