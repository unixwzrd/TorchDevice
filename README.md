# TorchDevice *WIP*`

TorchDevice is a class in the torchdevice.py package that intercepts PyTorch calls related to GPU hardware, enabling transparent code portability between NVIDIA CUDA and Apple Silicon (MPS) hardware. It allows developers to write code that works seamlessly on both hardware types without modification and is meant to assist in porting code from CUDA to MPS.

## Table of Contents

- [TorchDevice *WIP*\`](#torchdevice-wip)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Installing PyTorch](#installing-pytorch)
      - [Nightly builds for Apple Silicon](#nightly-builds-for-apple-silicon)
    - [Installing TorchDevice](#installing-torchdevice)
  - [Usage](#usage)
    - [Important Notes](#important-notes)
  - [Demo Scripts](#demo-scripts)
  - [Limitations](#limitations)
  - [Contributing](#contributing)
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
import torchdevice  # import torchdevice to apply patches
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

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss potential changes or improvements.

## License

```text
This project is licensed under the Apache License
                           Version 2.0, License.

 Copyright 2024 Michael P. Sullivan - unixwzrd@unixwzrd.ai

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
