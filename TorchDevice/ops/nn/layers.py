"""
TorchDevice Neural Network Layers Module
------------------------------------
This module patches torch.nn.Module methods like .to(), .cuda(), .mps(), .cpu()
to ensure they respect the device selection policy managed by DeviceManager.
"""

import torch
from typing import Any, Callable

from ...core.logger import log_info, auto_log
from ...core.device import DeviceManager

_original_module_to: Callable[..., Any] | None = None
_original_module_cuda: Callable[..., Any] | None = None
_original_module_mps: Callable[..., Any] | None = None
_original_module_cpu: Callable[..., Any] | None = None

@auto_log()
def _patched_module_to(module_instance: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.nn.Module:
    """Patched version of torch.nn.Module.to()"""
    global _original_module_to
    if not _original_module_to:
        # This should not happen if apply_patches was called correctly
        log_error("Original torch.nn.Module.to not found, calling unpatched version.")
        return module_instance # Or raise an error

    processed_args = list(args)
    device_arg_index = -1

    # Check positional arguments for device
    if args:
        # Common signatures: .to(device, dtype=None, non_blocking=False) or .to(other, ...)
        # We are interested in the first arg if it's a device specifier
        if isinstance(args[0], (str, torch.device)):
            device_arg_index = 0
            current_device_spec = args[0]
            resolved_device = DeviceManager.torch_device_replacement(current_device_spec)
            processed_args[0] = resolved_device
            # If device was positional, remove from kwargs to avoid conflict if it was also passed there
            kwargs.pop('device', None)

    # Check keyword arguments for device if not found positionally
    if device_arg_index == -1 and 'device' in kwargs:
        current_device_spec = kwargs['device']
        if isinstance(current_device_spec, (str, torch.device)):
            resolved_device = DeviceManager.torch_device_replacement(current_device_spec)
            kwargs['device'] = resolved_device

    return _original_module_to(module_instance, *processed_args, **kwargs)

@auto_log()
def _patched_module_cuda(module_instance: torch.nn.Module, device_arg: Any = None) -> torch.nn.Module:
    """Patched version of torch.nn.Module.cuda()"""
    global _original_module_to
    if not _original_module_to:
        log_error("Original torch.nn.Module.to not found for cuda patch.")
        return module_instance

    if device_arg is None:
        target_device_spec = 'cuda'
    elif isinstance(device_arg, int):
        target_device_spec = f'cuda:{device_arg}'
    elif isinstance(device_arg, torch.device) and device_arg.type == 'cuda':
        target_device_spec = device_arg
    else:
        # Pass through other types of device_arg (e.g. string 'cuda:1') for DeviceManager to handle
        target_device_spec = device_arg
    
    resolved_device = DeviceManager.torch_device_replacement(target_device_spec)
    return _original_module_to(module_instance, resolved_device)

@auto_log()
def _patched_module_mps(module_instance: torch.nn.Module) -> torch.nn.Module:
    """Patched version of torch.nn.Module.mps()"""
    global _original_module_to
    if not _original_module_to:
        log_error("Original torch.nn.Module.to not found for mps patch.")
        return module_instance
    
    resolved_device = DeviceManager.torch_device_replacement('mps')
    return _original_module_to(module_instance, resolved_device)

@auto_log()
def _patched_module_cpu(module_instance: torch.nn.Module) -> torch.nn.Module:
    """Patched version of torch.nn.Module.cpu()"""
    global _original_module_to
    if not _original_module_to:
        log_error("Original torch.nn.Module.to not found for cpu patch.")
        return module_instance

    resolved_device = DeviceManager.torch_device_replacement('cpu')
    return _original_module_to(module_instance, resolved_device)

def apply_patches() -> None:
    """Apply nn.Module method patches."""
    global _original_module_to, _original_module_cuda, _original_module_mps, _original_module_cpu
    
    log_info("Applying torch.nn.Module patches")

    if not hasattr(torch.nn.Module, '_is_torchdevice_patched_nn_layers'):
        _original_module_to = torch.nn.Module.to
        setattr(torch.nn.Module, 'to', _patched_module_to)

        if hasattr(torch.nn.Module, 'cuda'):
            _original_module_cuda = torch.nn.Module.cuda
            setattr(torch.nn.Module, 'cuda', _patched_module_cuda)
        else:
            log_info("torch.nn.Module.cuda not found, creating patch.")
            setattr(torch.nn.Module, 'cuda', _patched_module_cuda)  # Create if not exists

        if hasattr(torch.nn.Module, 'mps'):
            _original_module_mps = torch.nn.Module.mps
            setattr(torch.nn.Module, 'mps', _patched_module_mps)
        else:
            log_info("torch.nn.Module.mps not found, creating patch.")
            setattr(torch.nn.Module, 'mps', _patched_module_mps)  # Create if not exists

        if hasattr(torch.nn.Module, 'cpu'):
            _original_module_cpu = torch.nn.Module.cpu
            setattr(torch.nn.Module, 'cpu', _patched_module_cpu)
        else:
            # .cpu() should always exist on nn.Module
            log_info("torch.nn.Module.cpu not found (unexpected), creating patch.")
            setattr(torch.nn.Module, 'cpu', _patched_module_cpu)

        setattr(torch.nn.Module, '_is_torchdevice_patched_nn_layers', True)
        log_info("torch.nn.Module patches applied.")
    else:
        log_info("torch.nn.Module patches already applied.")

__all__ = ['apply_patches']

log_info("TorchDevice nn.layers module initialized and patches ready.")