"""
TorchDevice Tensor Creation Operations
--------------------------------
Tensor creation and manipulation operations.
"""

import torch
from typing import List, Callable, Any
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

    torch.Tensor.cpu = tensor_cpu_replacement
    torch.Tensor.numpy = numpy_replacement

@auto_log()
def tensor_cpu_replacement(tensor: torch.Tensor) -> torch.Tensor:
    """
    Replacement for torch.Tensor.cpu() that follows device redirection policy.
    If CPU override is active, moves to CPU, otherwise redirects to default device.
    """
    return tensor.to('cpu')

@auto_log()
def numpy_replacement(tensor: torch.Tensor) -> Any:
    """
    Replacement for torch.Tensor.numpy() that moves tensor to CPU first if needed.
    This always needs to go to CPU regardless of device policy since numpy()
    requires CPU tensors.
    """
    # Always move to CPU for numpy conversion - this is a special case
    # that must bypass the device redirection policy
    if tensor.device.type != 'cpu':
        cpu_tensor = tensor.to('cpu')
        return cpu_tensor.numpy()
    return tensor.numpy()

__all__: List[str] = [
    'apply_patches'
] 