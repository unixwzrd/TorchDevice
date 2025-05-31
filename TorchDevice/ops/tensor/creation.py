"""
TorchDevice Tensor Creation Operations
--------------------------------
Tensor creation and manipulation operations.
"""

import torch
from typing import List, Callable
from ...core.logger import log_info, auto_log
from ...core.device import (
    torch_device_replacement,
    t_Tensor_to,  # Import from core.device instead of redefining
    t_Tensor_numpy  # Also import numpy method from core
)

log_info("Importing TorchDevice/ops/tensor/creation.py")

# Flag to prevent recursion in tensor creation
_in_creation = False

def tensor_creation_wrapper(original_func: Callable) -> Callable:
    """
    Wrapper for tensor creation functions to enforce default device redirection and CPU override.
    Always calls torch_device_replacement for the device argument, so the toggle state is always respected.
    """
    @auto_log()
    def wrapped_func(*args, **kwargs):
        global _in_creation
        
        if _in_creation:
            # If we're already in a creation operation, use the original directly
            return original_func(*args, **kwargs)
        
        _in_creation = True
        try:
            device_arg = kwargs.get('device', None)
            if device_arg is None:
                device = torch_device_replacement()
                kwargs['device'] = device
            else:
                device = torch_device_replacement(device_arg)
                kwargs['device'] = device
            return original_func(*args, **kwargs)
        finally:
            _in_creation = False

    return wrapped_func

def apply_patches() -> None:
    """Apply tensor creation patches."""
    log_info("Applying tensor creation patches")

    # Patch tensor creation functions
    tensor_creation_functions = [
        'tensor', 'zeros', 'ones', 'empty', 'full',
        'arange', 'linspace', 'logspace', 'as_tensor',
        'empty_like', 'zeros_like', 'ones_like', 'full_like',
        'rand', 'randn', 'randint', 'rand_like', 'randn_like', 'randint_like'
    ]

    for func_name in tensor_creation_functions:
        if hasattr(torch, func_name):
            original_func = getattr(torch, func_name)
            patched_func = tensor_creation_wrapper(original_func)
            setattr(torch, func_name, patched_func)

    torch.Tensor.cpu = tensor_cpu_replacement
    torch.Tensor.numpy = numpy_replacement

@auto_log()
def tensor_cpu_replacement(tensor: torch.Tensor) -> torch.Tensor:
    """
    Replacement for torch.Tensor.cpu() that follows device redirection policy.
    If CPU override is active, moves to CPU, otherwise redirects to default device.
    """
    # Always use original to() method to move to CPU - this is a special case
    # that must bypass the device redirection policy
    return t_Tensor_to(tensor, 'cpu')

@auto_log()
def numpy_replacement(tensor: torch.Tensor):
    """
    Replacement for torch.Tensor.numpy() that moves tensor to CPU first if needed.
    This always needs to go to CPU regardless of device policy since numpy()
    requires CPU tensors.
    """
    # Always move to CPU for numpy conversion - this is a special case
    # that must bypass the device redirection policy
    if tensor.device.type != 'cpu':
        # First move to CPU using original to() method
        cpu_tensor = t_Tensor_to(tensor, 'cpu')
        return t_Tensor_numpy(cpu_tensor)
    return t_Tensor_numpy(tensor)

__all__: List[str] = [
    'apply_patches'
] 