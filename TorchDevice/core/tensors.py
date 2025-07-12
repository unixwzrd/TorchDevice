"""
TorchDevice Core Tensors Module
---------------------------
Core tensor movement and device redirection functionality.
"""

import torch
import threading
from typing import Any, Callable, TypeVar
from functools import wraps
from .logger import log_info, auto_log
from .device import DeviceManager, _ORIGINAL_TORCH_DEVICE_CLASS


# Original torch.Tensor function references - initialized during apply_patches
t_Tensor_to = None
t_Tensor_cuda = None
t_Tensor_cpu = None
t_Tensor_mps = None
t_Tensor_numpy = None  # For torch.Tensor.numpy()


@auto_log()
def tensor_to_replacement(tensor, *args, **kwargs):
    """Replacement for torch.Tensor.to() that follows device redirection policy."""
    if t_Tensor_to is None:
        raise RuntimeError("tensor_to_replacement called before patches applied")
        
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor_to_replacement called on non-tensor object: {type(tensor)}")
    if args and isinstance(args[0], (str, _ORIGINAL_TORCH_DEVICE_CLASS)):
        # First argument is device spec
        device = DeviceManager.torch_device_replacement(args[0])
        args = (device,) + args[1:]
        kwargs.pop('device', None)
    elif 'device' in kwargs:
        # Device specified in kwargs
        kwargs['device'] = DeviceManager.torch_device_replacement(kwargs['device'])
    return t_Tensor_to(tensor, *args, **kwargs)


@auto_log()
def tensor_cuda_replacement(tensor, device=None, non_blocking=False, memory_format=torch.preserve_format):
    """Replacement for torch.Tensor.cuda() that follows device redirection policy."""
    if t_Tensor_cuda is None:
        raise RuntimeError("tensor_cuda_replacement called before patches applied")
    
    device_spec = 'cuda' if device is None else f'cuda:{device}'
    device = DeviceManager.torch_device_replacement(device_spec)
    return t_Tensor_to(tensor, device, non_blocking=non_blocking, memory_format=memory_format)


@auto_log()
def tensor_cpu_replacement(tensor):
    """Replacement for torch.Tensor.cpu() that follows device redirection policy."""
    if t_Tensor_cpu is None:
        raise RuntimeError("tensor_cpu_replacement called before patches applied")
        
    device = DeviceManager.torch_device_replacement('cpu')
    return t_Tensor_to(tensor, device)


@auto_log()
def tensor_mps_replacement(tensor, non_blocking=False, memory_format=torch.preserve_format):
    """Replacement for torch.Tensor.mps() that follows device redirection policy."""
    if t_Tensor_mps is None:
        raise RuntimeError("tensor_mps_replacement called before patches applied")
    
    device = DeviceManager.torch_device_replacement('mps')
    return t_Tensor_to(tensor, device, non_blocking=non_blocking, memory_format=memory_format)


@auto_log()
def numpy_replacement(tensor):
    """
    Replacement for torch.Tensor.numpy() that moves tensor to CPU first if needed.
    This always needs to go to CPU regardless of device policy since numpy()
    requires CPU tensors.
    """
    if t_Tensor_numpy is None:
        raise RuntimeError("numpy_replacement called before patches applied or t_Tensor_numpy not initialized")

    # Always move to CPU for numpy conversion, bypassing redirection for this specific target.
    # We use the original t_Tensor_to to ensure it goes to 'cpu' without further redirection checks.
    if tensor.device.type != 'cpu':
        # Use the original t_Tensor_to to guarantee a direct move to CPU without DeviceManager interception
        # as this is a specific requirement for .numpy()
        if t_Tensor_to is None: # Should have been initialized by apply_patches
             raise RuntimeError("t_Tensor_to not initialized before numpy_replacement call")
        cpu_tensor = t_Tensor_to(tensor, 'cpu')
        return t_Tensor_numpy(cpu_tensor)
    return t_Tensor_numpy(tensor)


T = TypeVar('T')


@auto_log()
def tensor_creation_wrapper(original_func: Callable[..., T]) -> Callable[..., T]:
    """
    Wrapper for tensor creation functions to enforce default device redirection.
    The @auto_log decorator handles re-entry to prevent duplicate logs.
    """
    @auto_log()
    @wraps(original_func)
    def wrapped_func(*args: Any, **kwargs: Any) -> T:
        device_arg = kwargs.get('device', None)

        if device_arg is None:
            # No device specified, use the current default.
            resolved_device = DeviceManager.torch_device_replacement()
            kwargs['device'] = resolved_device
        else:
            # Device specified, resolve it through DeviceManager.
            resolved_device = DeviceManager.torch_device_replacement(device_arg)
            kwargs['device'] = resolved_device
        
        return original_func(*args, **kwargs)

    return wrapped_func


def apply_patches() -> None:
    """Apply torch.Tensor method patches."""
    log_info("Applying torch.Tensor method patches (to, cuda, cpu, mps, numpy)")
    
    global t_Tensor_to, t_Tensor_cuda, t_Tensor_cpu, t_Tensor_mps, t_Tensor_numpy
    
    if t_Tensor_to is None: # Check one, assume all tensor methods need init if this one does
        t_Tensor_to = torch.Tensor.to
        t_Tensor_cuda = torch.Tensor.cuda
        t_Tensor_cpu = torch.Tensor.cpu
        t_Tensor_numpy = torch.Tensor.numpy
        if hasattr(torch.Tensor, 'mps'):
            t_Tensor_mps = torch.Tensor.mps
        log_info("Original torch.Tensor methods stored")
    
    torch.Tensor.to = tensor_to_replacement
    torch.Tensor.cuda = tensor_cuda_replacement
    torch.Tensor.cpu = tensor_cpu_replacement
    torch.Tensor.numpy = numpy_replacement # Patch numpy
    
    if hasattr(torch.Tensor, 'mps') and t_Tensor_mps is not None:
        torch.Tensor.mps = tensor_mps_replacement
        
    log_info("torch.Tensor method patches applied")


# Module initialization
log_info("Initializing TorchDevice core tensors module")

__all__: list[str] = [
    'tensor_to_replacement',
    'tensor_cuda_replacement',
    'tensor_cpu_replacement',
    'tensor_mps_replacement',
    'numpy_replacement',
    'tensor_creation_wrapper',
    'apply_patches'
]

log_info("TorchDevice core tensors module initialized")
