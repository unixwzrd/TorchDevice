# Initial Package Structure Implementation Plan

## Current Implementation
The current implementation has everything in a single file: TorchDevice.py. This makes it difficult to maintain and extend.

## New Implementation

### 1. Create Package Structure

```bash
mkdir -p TorchDevice/TorchDevice/cuda
mkdir -p TorchDevice/TorchDevice/mps
mkdir -p TorchDevice/TorchDevice/tensor
mkdir -p TorchDevice/TorchDevice/module
touch TorchDevice/TorchDevice/__init__.py
touch TorchDevice/TorchDevice/core.py
touch TorchDevice/TorchDevice/logging.py
touch TorchDevice/TorchDevice/utils.py
touch TorchDevice/TorchDevice/cuda/__init__.py
touch TorchDevice/TorchDevice/cuda/mocks.py
touch TorchDevice/TorchDevice/cuda/events.py
touch TorchDevice/TorchDevice/cuda/streams.py
touch TorchDevice/TorchDevice/mps/__init__.py
touch TorchDevice/TorchDevice/mps/mocks.py
touch TorchDevice/TorchDevice/mps/events.py
touch TorchDevice/TorchDevice/mps/streams.py
touch TorchDevice/TorchDevice/tensor/__init__.py
touch TorchDevice/TorchDevice/tensor/operations.py
touch TorchDevice/TorchDevice/module/__init__.py
touch TorchDevice/TorchDevice/module/operations.py
```

### 2. Create __init__.py

```python
"""
TorchDevice - A library for transparently redirecting PyTorch CUDA calls to MPS or CPU.

This library hooks into PyTorch and redirects calls to the appropriate backend:
- If the hardware has CUDA, pass through the CUDA calls directly to PyTorch
- If the hardware has MPS, pass through the MPS calls directly to PyTorch
- If there is no CUDA or MPS available, fall back to CPU

Usage:
    import TorchDevice
    # That's it! All PyTorch calls will be redirected as needed.

Note:
    This library is never called directly, but hooks into PyTorch and redirects
    the calls to the appropriate location.
"""

import os
import sys
import torch

from .logging import (
    log_info, log_warning, log_error,
    set_verbosity, set_log_file, get_verbosity,
    LOG_VERBOSITY
)
from .core import TorchDevice

# Initialize TorchDevice when the module is imported
_torchdevice_instance = TorchDevice()
_torchdevice_instance.initialize()

# Export public API
__all__ = [
    'TorchDevice',
    'log_info', 'log_warning', 'log_error',
    'set_verbosity', 'set_log_file', 'get_verbosity'
]

# Version information
__version__ = '0.1.0'
```

### 3. Create core.py (Initial structure)

```python
"""
Core TorchDevice module.

This module contains the main TorchDevice class that handles initialization
and patching of PyTorch.
"""

import os
import sys
import torch
import inspect
import platform
from typing import Optional, Dict, Any, List, Tuple, Union

from .logging import log_info, log_warning, log_error, LOG_VERBOSITY
from .utils import get_default_device

# Import mock implementations
from .cuda.events import create_event as mock_cuda_event
from .cuda.streams import (
    create_stream as mock_cuda_stream_class,
    stream_context as mock_cuda_stream,
    current_stream as mock_cuda_current_stream,
    default_stream as mock_cuda_default_stream
)

# Import tensor and module operations
from .tensor.operations import (
    tensor_cuda_replacement,
    tensor_to_replacement
)
from .module.operations import (
    module_cuda_replacement,
    module_to_replacement
)

class TorchDevice:
    """
    Main TorchDevice class.
    
    This class handles initialization and patching of PyTorch to redirect
    calls to the appropriate backend.
    """
    
    # Class variables
    _initialized = False
    _default_device = None
    _original_cuda_is_available = None
    _original_mps_is_available = None
    _original_tensor_cuda = None
    _original_module_cuda = None
    _original_tensor_to = None
    _original_module_to = None
    _original_torch_load = None
    
    def __init__(self):
        """
        Initialize TorchDevice.
        """
        if LOG_VERBOSITY > 0:
            log_info("TorchDevice initializing", "TorchDevice.__init__")
    
    def initialize(self):
        """
        Initialize TorchDevice and apply patches.
        """
        if self._initialized:
            if LOG_VERBOSITY > 0:
                log_info("TorchDevice already initialized", "TorchDevice.initialize")
            return
        
        if LOG_VERBOSITY > 0:
            log_info("TorchDevice initializing", "TorchDevice.initialize")
        
        # Determine the default device
        self._default_device = get_default_device()
        
        # Save original functions
        self._original_cuda_is_available = torch.cuda.is_available
        self._original_tensor_cuda = torch.Tensor.cuda
        self._original_module_cuda = torch.nn.Module.cuda
        self._original_tensor_to = torch.Tensor.to
        self._original_module_to = torch.nn.Module.to
        self._original_torch_load = torch.load
        
        try:
            self._original_mps_is_available = torch.backends.mps.is_available
        except AttributeError:
            self._original_mps_is_available = lambda: False
        
        # Apply patches
        self.apply_patches()
        
        self._initialized = True
        if LOG_VERBOSITY > 0:
            log_info(f"TorchDevice initialized with default device: {self._default_device}", "TorchDevice.initialize")
    
    def apply_patches(self):
        """
        Apply patches to PyTorch functions.
        """
        if LOG_VERBOSITY > 0:
            log_info("Applying patches to PyTorch", "TorchDevice.apply_patches")
        
        # Patch tensor operations
        torch.Tensor.cuda = tensor_cuda_replacement
        torch.Tensor.to = tensor_to_replacement
        
        # Patch module operations
        torch.nn.Module.cuda = module_cuda_replacement
        torch.nn.Module.to = module_to_replacement
        
        # Patch torch.load
        torch.load = self.mock_torch_load
        
        # Patch CUDA functions if needed
        if self._default_device == 'mps':
            self.apply_cuda_patches()
        
        # Patch MPS functions if needed
        if self._default_device == 'cuda':
            self.apply_mps_patches()
    
    def apply_cuda_patches(self):
        """
        Apply patches to CUDA functions when running on MPS.
        """
        if LOG_VERBOSITY > 0:
            log_info("Applying CUDA patches", "TorchDevice.apply_cuda_patches")
        
        # Event and Stream related functions
        torch.cuda.Event = mock_cuda_event
        torch.cuda.Stream = mock_cuda_stream_class
        torch.cuda.stream = mock_cuda_stream
        torch.cuda.current_stream = mock_cuda_current_stream
        torch.cuda.default_stream = mock_cuda_default_stream
        
        # TODO: Add other CUDA patches from the original file
    
    def apply_mps_patches(self):
        """
        Apply patches to MPS functions when running on CUDA.
        """
        if LOG_VERBOSITY > 0:
            log_info("Applying MPS patches", "TorchDevice.apply_mps_patches")
        
        # TODO: Add MPS patches from the original file
    
    def mock_torch_load(self, *args, **kwargs):
        """
        Mock implementation of torch.load.
        """
        # TODO: Implement from the original file
        pass
```

### 4. Create utils.py

```python
"""
Utility functions for TorchDevice.
"""

import os
import torch
from typing import Optional, Dict, Any, List, Tuple, Union

from .logging import log_info, log_warning, log_error, LOG_VERBOSITY

def get_default_device() -> str:
    """
    Determine the default device to use.
    
    Returns:
        str: The default device ('cuda', 'mps', or 'cpu').
    """
    # Check environment variable first
    env_device = os.environ.get('TORCHDEVICE_DEFAULT_DEVICE', '').lower()
    if env_device in ['cuda', 'mps', 'cpu']:
        if LOG_VERBOSITY > 0:
            log_info(f"Using device from environment variable: {env_device}", "get_default_device")
        return env_device
    
    # Check available devices
    cuda_available = torch.cuda.is_available()
    
    try:
        mps_available = torch.backends.mps.is_available()
    except AttributeError:
        mps_available = False
    
    # Determine default device
    if cuda_available:
        default_device = 'cuda'
    elif mps_available:
        default_device = 'mps'
    else:
        default_device = 'cpu'
    
    if LOG_VERBOSITY > 0:
        log_info(f"Detected default device: {default_device}", "get_default_device")
    
    return default_device
```

### 5. Update setup.py

```python
from setuptools import setup, find_packages

setup(
    name="TorchDevice",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for transparently redirecting PyTorch CUDA calls to MPS or CPU",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TorchDevice",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
```

## Implementation Steps

1. Create the directory structure
2. Create the initial files with the basic structure
3. Implement the logging module first (as it's used by all other modules)
4. Implement the utils module
5. Implement the core module with basic functionality
6. Implement the cuda and mps modules
7. Implement the tensor and module operations
8. Update the package initialization
9. Test the refactored code

## Testing

After each module is implemented, we should run the tests to ensure that the functionality is preserved:

```bash
cd TorchDevice
tests/run_tests_and_install.py
```

## Benefits
- Cleaner organization of code
- Easier to maintain and extend
- Better separation of concerns
- Improved type hints and documentation
- More modular design
- Easier to test individual components
