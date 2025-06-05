"""
TorchDevice Core Tensors Module
---------------------------
Core tensor movement and device redirection functionality.
"""

import torch
from .logger import log_info, auto_log
from .device import DeviceManager, _ORIGINAL_TORCH_DEVICE_CLASS


# Original function references - initialized during apply_patches
t_Tensor_to = None
t_module_to = None
t_Tensor_cuda = None
t_module_cuda = None
t_Tensor_cpu = None
t_module_cpu = None
t_Tensor_mps = None
t_module_mps = None


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
def module_to_replacement(module, *args, **kwargs):
    """Replacement for torch.nn.Module.to() that follows device redirection policy."""
    if t_module_to is None:
        raise RuntimeError("module_to_replacement called before patches applied")
        
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"module_to_replacement called on non-module object: {type(module)}")
    if args and isinstance(args[0], (str, _ORIGINAL_TORCH_DEVICE_CLASS)):
        # First argument is device spec
        device = DeviceManager.torch_device_replacement(args[0])
        args = (device,) + args[1:]
        kwargs.pop('device', None)
    elif 'device' in kwargs:
        # Device specified in kwargs
        kwargs['device'] = DeviceManager.torch_device_replacement(kwargs['device'])
    return t_module_to(module, *args, **kwargs)


@auto_log()
def tensor_cuda_replacement(tensor, device=None, non_blocking=False, memory_format=torch.preserve_format):
    """Replacement for torch.Tensor.cuda() that follows device redirection policy."""
    if t_Tensor_cuda is None:
        raise RuntimeError("tensor_cuda_replacement called before patches applied")
    
    device_spec = 'cuda' if device is None else f'cuda:{device}'
    device = DeviceManager.torch_device_replacement(device_spec)
    return t_Tensor_to(tensor, device, non_blocking=non_blocking, memory_format=memory_format)


@auto_log()
def module_cuda_replacement(module, device=None):
    """Replacement for torch.nn.Module.cuda() that follows device redirection policy."""
    if t_module_cuda is None:
        raise RuntimeError("module_cuda_replacement called before patches applied")
        
    device_spec = 'cuda' if device is None else f'cuda:{device}'
    device = DeviceManager.torch_device_replacement(device_spec)
    return t_module_to(module, device)


@auto_log()
def tensor_cpu_replacement(tensor):
    """Replacement for torch.Tensor.cpu() that follows device redirection policy."""
    if t_Tensor_cpu is None:
        raise RuntimeError("tensor_cpu_replacement called before patches applied")
        
    device = DeviceManager.torch_device_replacement('cpu')
    return t_Tensor_to(tensor, device)


@auto_log()
def module_cpu_replacement(module):
    """Replacement for torch.nn.Module.cpu() that follows device redirection policy."""
    if t_module_cpu is None:
        raise RuntimeError("module_cpu_replacement called before patches applied")
        
    device = DeviceManager.torch_device_replacement('cpu')
    return t_module_to(module, device)


@auto_log()
def tensor_mps_replacement(tensor, non_blocking=False, memory_format=torch.preserve_format):
    """Replacement for torch.Tensor.mps() that follows device redirection policy."""
    if t_Tensor_mps is None:
        raise RuntimeError("tensor_mps_replacement called before patches applied")
    
    device = DeviceManager.torch_device_replacement('mps')
    return t_Tensor_to(tensor, device, non_blocking=non_blocking, memory_format=memory_format)


@auto_log()
def module_mps_replacement(module):
    """Replacement for torch.nn.Module.mps() that follows device redirection policy."""
    if t_module_mps is None:
        raise RuntimeError("module_mps_replacement called before patches applied")
        
    device = DeviceManager.torch_device_replacement('mps')
    return t_module_to(module, device)


def apply_patches() -> None:
    """Apply tensor movement patches."""
    log_info("Applying tensor movement patches")
    
    # Store original functions FIRST, before any patching
    # This avoids storing already-patched functions
    global t_Tensor_to, t_module_to, t_Tensor_cuda, t_module_cuda, t_Tensor_cpu, t_module_cpu, t_Tensor_mps, t_module_mps
    
    # Only store originals if not already initialized
    if t_Tensor_to is None:
        t_Tensor_to = torch.Tensor.to
        t_module_to = torch.nn.Module.to
        t_Tensor_cuda = torch.Tensor.cuda
        t_module_cuda = torch.nn.Module.cuda
        t_Tensor_cpu = torch.Tensor.cpu
        t_module_cpu = torch.nn.Module.cpu
        
        # Store MPS methods if they exist
        if hasattr(torch.Tensor, 'mps'):
            t_Tensor_mps = torch.Tensor.mps
        if hasattr(torch.nn.Module, 'mps'):
            t_module_mps = torch.nn.Module.mps
        
        log_info("Original tensor functions stored")
    
    # Replace with our versions
    torch.Tensor.to = tensor_to_replacement
    torch.nn.Module.to = module_to_replacement
    torch.Tensor.cuda = tensor_cuda_replacement
    torch.nn.Module.cuda = module_cuda_replacement
    torch.Tensor.cpu = tensor_cpu_replacement
    torch.nn.Module.cpu = module_cpu_replacement
    
    # Patch MPS methods if they exist
    if hasattr(torch.Tensor, 'mps') and t_Tensor_mps is not None:
        torch.Tensor.mps = tensor_mps_replacement
    if hasattr(torch.nn.Module, 'mps') and t_module_mps is not None:
        torch.nn.Module.mps = module_mps_replacement
    
    log_info("Tensor movement patches applied")


# Module initialization
log_info("Initializing TorchDevice core tensors module")

__all__: list[str] = [
    'tensor_to_replacement',
    'module_to_replacement',
    'tensor_cuda_replacement',
    'module_cuda_replacement',
    'tensor_cpu_replacement',
    'module_cpu_replacement',
    'tensor_mps_replacement',
    'module_mps_replacement',
    'apply_patches'
]

log_info("TorchDevice core tensors module initialized")