# TorchDevice

![TorchDevice](docs/TorchDevice001.png)

A Python library for transparent device handling in PyTorch, enabling seamless transitions between CUDA and MPS devices.

## Overview

TorchDevice provides transparent device handling for PyTorch applications, allowing code written for CUDA to run on MPS devices (and vice versa) without modification. It intercepts device-specific calls and redirects them appropriately based on available hardware. This is primarily for migration of existing CUDA code and running existing CUDA code on Apple Silicon Macs.

## Features

- **Transparent Device Handling**: Automatically redirects CUDA calls to MPS (and vice versa)
- **Comprehensive PyTorch Integration**: 
  - Neural Network Operations
  - Memory Management
  - Stream and Event Handling
  - Automatic Differentiation
  - Optimization Algorithms
- **Robust Error Handling**: Graceful fallbacks and informative error messages
- **Performance Optimization**: Efficient device-specific implementations
- **Type Safety**: Comprehensive type checking and validation

## Installation

Clone and install from the repository:
```bash
git clone https://github.com/unixwzrd/TorchDevice.git
cd TorchDevice
pip install -e .
```

## Quick Start

```python
import TorchDevice  # This automatically hooks into PyTorch, order not important.
import torch

# Your existing PyTorch code works as is
model = torch.nn.Linear(10, 10).cuda()  # Will use MPS if CUDA isn't available
```

**NOTE:** The project is under active development and produces very verbose logs, so you may want to redirect them to a file or `/dev/null`. I will be working on reducing the verbosity in the future.

## Project Structure

The project is organized into three main components:

1. **Core** (`TorchDevice/core/`): Central device handling and patching
2. **Operations** (`TorchDevice/ops/`): Device-specific implementations
3. **Utilities** (`TorchDevice/utils/`): Shared functionality

For detailed structure information, see [Project Structure](docs/project_structure.md).

## Documentation

- [Project Structure](docs/project_structure.md): Detailed project organization
- [TorchDevice Behavior](docs/TorchDevice_Behavior.md): Core functionality and behavior
- [API Reference](docs/TorchDevice_Functions.md): Comprehensive API documentation
- [CUDA Operations](docs/CUDA-Operations.md): CUDA-specific functionality
- [Advanced External ProjectTesting](test_automation/README.md): Comprehensive integration testing

## Development

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- For CUDA development: NVIDIA CUDA Toolkit
- For MPS development: Apple Silicon Mac

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/TorchDevice.git
cd TorchDevice

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e  .
```

### Running Tests

```bash
# Run all tests
python run_tests_and_install.py

# Run specific test module
python run_tests_and_install.py --test-only tests/[test-name.py]
```

### Advanced Integration Testing

The most current project files are in the dev branch and may be accessed from there by checking out the project and then checking out the dev branch.

```bash
# Checkout the project
git clone https://github.com/yourusername/TorchDevice.git

cd TorchDevice

# Checkout the dev branch
git checkout dev
```

If you wish to contribute create a branch and then submit a PR for your changes. I will review them and merge them into the dev branch and then into the main branch as I get to them. I will try to get to them as soon as possible. I appreciate all contributions to making this better.

For comprehensive integration testing against large codebases like Hugging Face Transformers, a dedicated test automation suite is available. This suite manages complex test execution, logging, and reporting. For detailed instructions on setup, dependencies, and usage, please see the dedicated guide:

[**Advanced Test Automation README**](test_automation/README.md)

The latest test reports are in the [test_automation/reports](test_automation/reports) directory. Feel free to have a look and let me know if you have any questions or suggestions.  If you have any fixes, please submit a PR.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

It has additional information on testing other projects with TorchDevice, though I have mostly tested and am working on debugging the issues with HuggingFace Transformers. For now, I am focused on getting the core functionality working.  Right now about 13% of the tests are failing and many of those are due to CUDA specific functionality in the CUDA library.

This is not my primary project I am working on, but I am using it for other projects I am working on, giving me access to available PyTorch code written for CUDA and saving time in having to port code over to Apple Silicon.

## Supporting This Project

TorchDevice is an open-source project developed and maintained by [me, M S - unixwzrd](https://unixwzrd.ai). It has evolved and grown over the past 9 months from when it was first conceived, and is now relatively stable and useful for running PyTorch CUDA code on Apple Silicon, and I have tested it with a number of projects. Some projects having to do with Text-to-Speech and other AI/ML projects. My goal right now is to have the HuggingFace Transformers library working with it as much as possible.

If you find this project useful, please consider supporting its development:

### Funding Options
- [Patreon](https://www.patreon.com/unixwzrd)
- [Ko-Fi](https://ko-fi.com/unixwzrd)
- [Buy Me a Coffee](https://www.buymeacoffee.com/unixwzrd)

### Other Ways to Support
1. **Star the Repository**: Show your support by starring the project on GitHub
2. **Share the Project**: Help spread the word about TorchDevice
3. **Report Issues**: Help improve the project by reporting bugs or suggesting features
4. **Contribute Code**: Submit pull requests to help fix issues or add features
5. **Improve Documentation**: Help keep our documentation clear, accurate, and helpful

For more information about the developer and other projects, visit [Distributed Thinking Systems](https://unixwzrd.ai).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.