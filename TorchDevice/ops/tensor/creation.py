"""
TorchDevice Tensor Creation Module
----------------------------
Tensor creation utilities.
"""

import torch
from typing import List, Callable
from ...core.logger import log_info, auto_log
from ...core.device import torch_device_replacement

log_info("Importing TorchDevice/ops/tensor/creation.py")

def tensor_creation_wrapper(original_func: Callable) -> Callable:
    """
    Wrapper for tensor creation functions to enforce default device redirection and CPU override.
    Always calls torch_device_replacement for the device argument, so the toggle state is always respected.
    """
    @auto_log()
    def wrapped_func(*args, **kwargs):
        device_arg = kwargs.get('device', None)
        if device_arg is None:
            device = torch_device_replacement()
            kwargs['device'] = device
        else:
            device = torch_device_replacement(device_arg)
            kwargs['device'] = device
        return original_func(*args, **kwargs)

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

__all__: List[str] = [
    'apply_patches'
] 