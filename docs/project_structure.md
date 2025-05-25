# TorchDevice Project Structure

## Overview
TorchDevice is organized into a modular structure that mirrors PyTorch's architecture while providing transparent device handling and redirection. The project is divided into three main components:

1. Core (core/)
2. Operations (ops/)
3. Utilities (utils/)

## Directory Structure

```
TorchDevice/
├── core/                   # Core functionality
│   ├── device.py          # Main device handling and redirection
│   ├── patch.py           # PyTorch patching mechanism
│   └── logger.py          # Logging infrastructure
│
├── ops/                   # Operation-specific implementations
│   ├── memory/           # Memory management
│   │   ├── management.py # Memory allocation and tracking
│   │   └── stats.py      # Memory statistics and monitoring
│   │
│   ├── nn/               # Neural network operations
│   │   ├── containers.py # Container modules (Sequential, ModuleList, etc.)
│   │   ├── layers.py     # Basic layer implementations
│   │   ├── normalization.py # Normalization layers
│   │   ├── activation.py # Activation functions
│   │   ├── attention.py  # Attention mechanisms
│   │   └── init.py       # Weight initialization
│   │
│   ├── random/           # Random number generation
│   │   ├── generators.py # RNG implementations
│   │   └── distributions.py # Probability distributions
│   │
│   ├── streams/          # CUDA/MPS stream handling
│   │   ├── cuda.py       # CUDA stream implementations
│   │   ├── mps.py        # MPS stream implementations
│   │   └── synchronize.py # Stream synchronization utilities
│   │
│   ├── events/           # Event handling and synchronization
│   │   ├── cuda_events.py # CUDA event handling
│   │   ├── mps_events.py  # MPS event handling
│   │   └── synchronize.py # Event synchronization utilities
│   │
│   ├── autograd/         # Automatic differentiation
│   │   ├── function.py   # Custom autograd functions
│   │   ├── variable.py   # Variable handling
│   │   └── grad_mode.py  # Gradient mode context managers
│   │
│   └── optim/           # Optimization algorithms
│       ├── optimizer.py  # Base optimizer implementations
│       └── lr_scheduler.py # Learning rate scheduling
│
└── utils/               # Utility functions and helpers
    ├── compile.py      # Compilation utilities
    ├── profiling.py    # Performance profiling tools
    ├── type_utils.py   # Type checking and conversion
    ├── device_utils.py # Device-related utilities
    └── error_handling.py # Custom exceptions and error handling

```

## Component Responsibilities

### Core
- **device.py**: Central device management, handles device detection and selection
- **patch.py**: Manages PyTorch function patching and interception
- **logger.py**: Logging system for debugging and monitoring

### Operations (ops)
- **memory/**: Memory management and monitoring
- **nn/**: Neural network operations and layers
- **random/**: Random number generation and distributions
- **streams/**: CUDA and MPS stream management
- **events/**: Event handling and synchronization
- **autograd/**: Automatic differentiation support
- **optim/**: Optimization algorithms and learning rate scheduling

### Utils
- **compile.py**: Compilation and JIT-related utilities
- **profiling.py**: Performance profiling tools
- **type_utils.py**: Type checking and conversion utilities
- **device_utils.py**: Device-related helper functions
- **error_handling.py**: Custom exceptions and error handling

## Import Guidelines

1. Core imports should be direct:
```python
from TorchDevice.core import device
```

2. Operation imports should specify the submodule:
```python
from TorchDevice.ops.nn import layers
from TorchDevice.ops.memory import management
```

3. Utility imports can be direct or from submodules:
```python
from TorchDevice.utils import type_utils
```

## Development Guidelines

1. Keep core functionality minimal and focused on device handling
2. Place device-specific implementations in appropriate ops/ submodules
3. Use utils/ for shared functionality across modules
4. Maintain parallel structure in tests/ directory
5. Update __init__.py files when adding new modules
6. Document all public APIs using docstrings
7. Include type hints for all function parameters and return values 