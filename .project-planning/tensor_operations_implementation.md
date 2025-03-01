# Tensor Operations Module Implementation Plan

## Current Implementation

The current implementation includes:
- Replacements for `torch.Tensor.cuda()` method
- Replacements for `torch.Tensor.to()` method
- Various helper functions for tensor operations

## New Implementation

### 1. Create tensor/operations.py

```python
"""
Tensor operations module.

This module contains replacements for tensor operations like cuda() and to().
"""

import torch
import inspect
from typing import Optional, Dict, Any, List, Tuple, Union

from ..logging import log_info, log_warning, log_error, LOG_VERBOSITY
from ..utils import get_default_device

def tensor_cuda_replacement(self, *args, **kwargs):
    """
    Replacement for torch.Tensor.cuda() method.
    
    Args:
        self: The tensor.
        *args: Positional arguments.
        **kwargs: Keyword arguments.
    
    Returns:
        torch.Tensor: The tensor on the appropriate device.
    """
    # Get caller information for logging
    caller_info = {}
    try:
        frame = inspect.currentframe()
        if frame:
            frame = frame.f_back
            if frame:
                caller_info['file'] = frame.f_code.co_filename
                caller_info['line'] = frame.f_lineno
                caller_info['function'] = frame.f_code.co_name
    except Exception as e:
        if LOG_VERBOSITY > 0:
            log_warning(f"Error getting caller info: {e}", "torch.Tensor.cuda")
    
    # Determine the default device
    default_device = get_default_device()
    
    # Log the call
    if LOG_VERBOSITY > 0:
        log_info(f"torch.Tensor.cuda() called with args={args}, kwargs={kwargs}", "torch.Tensor.cuda")
        if caller_info:
            log_info(f"Called from {caller_info.get('file', 'unknown')}:{caller_info.get('line', 'unknown')} in {caller_info.get('function', 'unknown')}", "torch.Tensor.cuda")
    
    # Handle different device scenarios
    if default_device == 'cuda':
        # If we're on CUDA, just pass through to the original cuda method
        original_cuda = torch.Tensor.cuda.__wrapped__
        return original_cuda(self, *args, **kwargs)
    elif default_device == 'mps':
        # If we're on MPS, redirect to MPS device
        if LOG_VERBOSITY > 0:
            log_warning(f"Redirecting torch.Tensor.cuda() to MPS device", "torch.Tensor.cuda")
        
        # Convert args and kwargs for to() method
        to_args = []
        to_kwargs = {}
        
        # Handle device argument
        if 'device' in kwargs:
            # If device is explicitly specified, warn but respect it
            device = kwargs['device']
            if isinstance(device, torch.device) and device.type == 'cuda':
                to_kwargs['device'] = torch.device('mps')
            elif isinstance(device, str) and device.startswith('cuda'):
                to_kwargs['device'] = 'mps'
            else:
                to_kwargs['device'] = device
        else:
            to_kwargs['device'] = 'mps'
        
        # Handle other arguments
        if 'non_blocking' in kwargs:
            to_kwargs['non_blocking'] = kwargs['non_blocking']
        if 'memory_format' in kwargs:
            to_kwargs['memory_format'] = kwargs['memory_format']
        
        # Call to() with the modified arguments
        return self.to(*to_args, **to_kwargs)
    else:
        # If we're on CPU, just return the tensor on CPU
        if LOG_VERBOSITY > 0:
            log_warning(f"Redirecting torch.Tensor.cuda() to CPU device", "torch.Tensor.cuda")
        return self.cpu()

def tensor_to_replacement(self, *args, **kwargs):
    """
    Replacement for torch.Tensor.to() method.
    
    Args:
        self: The tensor.
        *args: Positional arguments.
        **kwargs: Keyword arguments.
    
    Returns:
        torch.Tensor: The tensor on the appropriate device.
    """
    # Get caller information for logging
    caller_info = {}
    try:
        frame = inspect.currentframe()
        if frame:
            frame = frame.f_back
            if frame:
                caller_info['file'] = frame.f_code.co_filename
                caller_info['line'] = frame.f_lineno
                caller_info['function'] = frame.f_code.co_name
    except Exception as e:
        if LOG_VERBOSITY > 0:
            log_warning(f"Error getting caller info: {e}", "torch.Tensor.to")
    
    # Determine the default device
    default_device = get_default_device()
    
    # Log the call
    if LOG_VERBOSITY > 0:
        log_info(f"torch.Tensor.to() called with args={args}, kwargs={kwargs}", "torch.Tensor.to")
        if caller_info:
            log_info(f"Called from {caller_info.get('file', 'unknown')}:{caller_info.get('line', 'unknown')} in {caller_info.get('function', 'unknown')}", "torch.Tensor.to")
    
    # Process arguments to determine the target device
    target_device = None
    
    # Check for device in args
    for arg in args:
        if isinstance(arg, torch.device):
            target_device = arg
            break
        elif isinstance(arg, str) and arg in ['cuda', 'cpu', 'mps']:
            target_device = arg
            break
    
    # Check for device in kwargs
    if 'device' in kwargs:
        target_device = kwargs['device']
    
    # If no device specified, just pass through to the original to method
    if target_device is None:
        original_to = torch.Tensor.to.__wrapped__
        return original_to(self, *args, **kwargs)
    
    # Handle different device scenarios
    if isinstance(target_device, torch.device):
        device_type = target_device.type
    elif isinstance(target_device, str):
        device_type = target_device.split(':')[0]
    else:
        # If we can't determine the device type, just pass through
        original_to = torch.Tensor.to.__wrapped__
        return original_to(self, *args, **kwargs)
    
    # Handle CUDA to MPS redirection
    if device_type == 'cuda' and default_device == 'mps':
        if LOG_VERBOSITY > 0:
            log_warning(f"Redirecting torch.Tensor.to(cuda) to MPS device", "torch.Tensor.to")
        
        # Create new args and kwargs with the device replaced
        new_args = list(args)
        new_kwargs = dict(kwargs)
        
        # Replace the device in args
        for i, arg in enumerate(new_args):
            if isinstance(arg, torch.device) and arg.type == 'cuda':
                new_args[i] = torch.device('mps')
            elif isinstance(arg, str) and arg.startswith('cuda'):
                new_args[i] = 'mps'
        
        # Replace the device in kwargs
        if 'device' in new_kwargs:
            device = new_kwargs['device']
            if isinstance(device, torch.device) and device.type == 'cuda':
                new_kwargs['device'] = torch.device('mps')
            elif isinstance(device, str) and device.startswith('cuda'):
                new_kwargs['device'] = 'mps'
        
        # Call the original to method with the modified arguments
        original_to = torch.Tensor.to.__wrapped__
        return original_to(self, *new_args, **new_kwargs)
    
    # Handle MPS to CUDA redirection
    elif device_type == 'mps' and default_device == 'cuda':
        if LOG_VERBOSITY > 0:
            log_warning(f"Redirecting torch.Tensor.to(mps) to CUDA device", "torch.Tensor.to")
        
        # Create new args and kwargs with the device replaced
        new_args = list(args)
        new_kwargs = dict(kwargs)
        
        # Replace the device in args
        for i, arg in enumerate(new_args):
            if isinstance(arg, torch.device) and arg.type == 'mps':
                new_args[i] = torch.device('cuda')
            elif isinstance(arg, str) and arg == 'mps':
                new_args[i] = 'cuda'
        
        # Replace the device in kwargs
        if 'device' in new_kwargs:
            device = new_kwargs['device']
            if isinstance(device, torch.device) and device.type == 'mps':
                new_kwargs['device'] = torch.device('cuda')
            elif isinstance(device, str) and device == 'mps':
                new_kwargs['device'] = 'cuda'
        
        # Call the original to method with the modified arguments
        original_to = torch.Tensor.to.__wrapped__
        return original_to(self, *new_args, **new_kwargs)
    
    # If no redirection needed, just pass through
    original_to = torch.Tensor.to.__wrapped__
    return original_to(self, *args, **kwargs)

def tensor_mps_replacement(self, *args, **kwargs):
    """
    Replacement for torch.Tensor.mps() method (if it exists).
    
    Args:
        self: The tensor.
        *args: Positional arguments.
        **kwargs: Keyword arguments.
    
    Returns:
        torch.Tensor: The tensor on the appropriate device.
    """
    # Get caller information for logging
    caller_info = {}
    try:
        frame = inspect.currentframe()
        if frame:
            frame = frame.f_back
            if frame:
                caller_info['file'] = frame.f_code.co_filename
                caller_info['line'] = frame.f_lineno
                caller_info['function'] = frame.f_code.co_name
    except Exception as e:
        if LOG_VERBOSITY > 0:
            log_warning(f"Error getting caller info: {e}", "torch.Tensor.mps")
    
    # Determine the default device
    default_device = get_default_device()
    
    # Log the call
    if LOG_VERBOSITY > 0:
        log_info(f"torch.Tensor.mps() called with args={args}, kwargs={kwargs}", "torch.Tensor.mps")
        if caller_info:
            log_info(f"Called from {caller_info.get('file', 'unknown')}:{caller_info.get('line', 'unknown')} in {caller_info.get('function', 'unknown')}", "torch.Tensor.mps")
    
    # Handle different device scenarios
    if default_device == 'mps':
        # If we're on MPS, just pass through to the original mps method
        original_mps = torch.Tensor.mps.__wrapped__
        return original_mps(self, *args, **kwargs)
    elif default_device == 'cuda':
        # If we're on CUDA, redirect to CUDA device
        if LOG_VERBOSITY > 0:
            log_warning(f"Redirecting torch.Tensor.mps() to CUDA device", "torch.Tensor.mps")
        
        # Convert args and kwargs for to() method
        to_args = []
        to_kwargs = {}
        
        # Handle device argument
        if 'device' in kwargs:
            # If device is explicitly specified, warn but respect it
            device = kwargs['device']
            if isinstance(device, torch.device) and device.type == 'mps':
                to_kwargs['device'] = torch.device('cuda')
            elif isinstance(device, str) and device == 'mps':
                to_kwargs['device'] = 'cuda'
            else:
                to_kwargs['device'] = device
        else:
            to_kwargs['device'] = 'cuda'
        
        # Handle other arguments
        if 'non_blocking' in kwargs:
            to_kwargs['non_blocking'] = kwargs['non_blocking']
        if 'memory_format' in kwargs:
            to_kwargs['memory_format'] = kwargs['memory_format']
        
        # Call to() with the modified arguments
        return self.to(*to_args, **to_kwargs)
    else:
        # If we're on CPU, just return the tensor on CPU
        if LOG_VERBOSITY > 0:
            log_warning(f"Redirecting torch.Tensor.mps() to CPU device", "torch.Tensor.mps")
        return self.cpu()

def patch_tensor_methods():
    """
    Patch tensor methods with our replacements.
    """
    # Save original methods
    if not hasattr(torch.Tensor.cuda, '__wrapped__'):
        torch.Tensor.cuda.__wrapped__ = torch.Tensor.cuda
    
    if not hasattr(torch.Tensor.to, '__wrapped__'):
        torch.Tensor.to.__wrapped__ = torch.Tensor.to
    
    # Check if mps method exists
    if hasattr(torch.Tensor, 'mps') and not hasattr(torch.Tensor.mps, '__wrapped__'):
        torch.Tensor.mps.__wrapped__ = torch.Tensor.mps
    
    # Apply patches
    torch.Tensor.cuda = tensor_cuda_replacement
    torch.Tensor.to = tensor_to_replacement
    
    if hasattr(torch.Tensor, 'mps'):
        torch.Tensor.mps = tensor_mps_replacement
    
    if LOG_VERBOSITY > 0:
        log_info("Tensor methods patched", "patch_tensor_methods")

def restore_tensor_methods():
    """
    Restore original tensor methods.
    """
    # Restore original methods if they were saved
    if hasattr(torch.Tensor.cuda, '__wrapped__'):
        torch.Tensor.cuda = torch.Tensor.cuda.__wrapped__
    
    if hasattr(torch.Tensor.to, '__wrapped__'):
        torch.Tensor.to = torch.Tensor.to.__wrapped__
    
    if hasattr(torch.Tensor, 'mps') and hasattr(torch.Tensor.mps, '__wrapped__'):
        torch.Tensor.mps = torch.Tensor.mps.__wrapped__
    
    if LOG_VERBOSITY > 0:
        log_info("Tensor methods restored", "restore_tensor_methods")
```

### 2. Create tensor/__init__.py

```python
"""
Tensor operations module.
"""

from .operations import (
    tensor_cuda_replacement,
    tensor_to_replacement,
    tensor_mps_replacement,
    patch_tensor_methods,
    restore_tensor_methods
)

__all__ = [
    'tensor_cuda_replacement',
    'tensor_to_replacement',
    'tensor_mps_replacement',
    'patch_tensor_methods',
    'restore_tensor_methods'
]
```

### 3. Update core.py to use the tensor operations module

```python
from .tensor.operations import patch_tensor_methods, restore_tensor_methods

def apply_patches(self):
    """
    Apply patches to PyTorch functions.
    """
    if LOG_VERBOSITY > 0:
        log_info("Applying patches to PyTorch", "TorchDevice.apply_patches")
    
    # Patch tensor operations
    patch_tensor_methods()
    
    # ... other patches ...
```

## Benefits
- Cleaner organization of code
- Easier to maintain and extend
- Better separation of concerns
- Improved type hints and documentation
- More modular design
- Easier to test individual components
