"""
TorchDevice Core nn.Module Methods Module
---------------------------------------
Functionality for patching and redirecting torch.nn.Module methods like .to(), .cuda(), etc.
"""

import torch
from .logger import log_info, auto_log
from .device import DeviceManager, _ORIGINAL_TORCH_DEVICE_CLASS

# Original torch.nn.Module function references - initialized during apply_patches
t_module_to = None
t_module_cuda = None
t_module_cpu = None
t_module_mps = None

@auto_log()
def module_to_replacement(module, *args, **kwargs):
    """Replacement for torch.nn.Module.to() that follows device redirection policy."""
    if t_module_to is None:
        raise RuntimeError("module_to_replacement called before patches applied")
        
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"module_to_replacement called on non-module object: {type(module)}")
    
    # DeviceManager.torch_device_replacement will handle None, str, or torch.device inputs
    if args and isinstance(args[0], (str, _ORIGINAL_TORCH_DEVICE_CLASS, int)):
        # First argument is device spec (str, int for index, or actual device object)
        device = DeviceManager.torch_device_replacement(args[0])
        args = (device,) + args[1:]
        kwargs.pop('device', None) # Remove device from kwargs if it was also passed positionally
    elif 'device' in kwargs:
        # Device specified in kwargs
        kwargs['device'] = DeviceManager.torch_device_replacement(kwargs['device'])
    
    return t_module_to(module, *args, **kwargs)

@auto_log()
def module_cuda_replacement(module, device=None):
    """Replacement for torch.nn.Module.cuda() that follows device redirection policy."""
    if t_module_cuda is None:
        raise RuntimeError("module_cuda_replacement called before patches applied")
        
    device_spec = 'cuda' if device is None else f'cuda:{device}'
    # Use module_to_replacement which correctly calls DeviceManager.torch_device_replacement
    # This avoids calling DeviceManager directly here and keeps logic centralized in _to_replacement
    return module_to_replacement(module, device_spec)

@auto_log()
def module_cpu_replacement(module):
    """Replacement for torch.nn.Module.cpu() that follows device redirection policy."""
    if t_module_cpu is None:
        raise RuntimeError("module_cpu_replacement called before patches applied")
        
    # Use module_to_replacement for consistency
    return module_to_replacement(module, 'cpu')

@auto_log()
def module_mps_replacement(module):
    """Replacement for torch.nn.Module.mps() that follows device redirection policy."""
    if t_module_mps is None:
        raise RuntimeError("module_mps_replacement called before patches applied")
    
    # Use module_to_replacement for consistency
    return module_to_replacement(module, 'mps')

def apply_patches() -> None:
    """Apply nn.Module method patches."""
    log_info("Applying nn.Module method patches")
    
    global t_module_to, t_module_cuda, t_module_cpu, t_module_mps
    
    if t_module_to is None: # Check one, assume all need init if this one does
        t_module_to = torch.nn.Module.to
        t_module_cuda = torch.nn.Module.cuda
        t_module_cpu = torch.nn.Module.cpu
        if hasattr(torch.nn.Module, 'mps'):
            t_module_mps = torch.nn.Module.mps
        log_info("Original nn.Module methods stored")
    
    torch.nn.Module.to = module_to_replacement
    torch.nn.Module.cuda = module_cuda_replacement
    torch.nn.Module.cpu = module_cpu_replacement
    if hasattr(torch.nn.Module, 'mps') and t_module_mps is not None:
        torch.nn.Module.mps = module_mps_replacement
    
    log_info("nn.Module method patches applied")

log_info("Initializing TorchDevice core nn.Module methods module")

__all__: list[str] = [
    'module_to_replacement',
    'module_cuda_replacement',
    'module_cpu_replacement',
    'module_mps_replacement',
    'apply_patches'
]

log_info("TorchDevice core nn.Module methods module initialized")
