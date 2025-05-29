"""
TorchDevice Core Device Module
--------------------------
Core device management and conversion functionality.
"""

import torch
import threading
from typing import Optional, Any, List, Union
from .logger import log_info, auto_log

# Store original device type and constructor
T_DEVICE_TYPE = torch.device("cpu").__class__
t_device_constructor = torch.device

# Global state
_default_device: Optional[torch.device] = None
_default_device_type: str = ""
_previous_default_device_type: str = ""
_device_lock = threading.RLock()
_cpu_override: bool = False
_patches_applied: bool = False

# Store original tensor conversion methods
t_Tensor_to = torch.Tensor.to
t_nn_Module_to = torch.nn.Module.to
t_Tensor_cpu = torch.Tensor.cpu
t_nn_Module_cpu = torch.nn.Module.cpu
t_Tensor_cuda = torch.Tensor.cuda
t_nn_Module_cuda = torch.nn.Module.cuda
t_Tensor_mps = torch.Tensor.mps if hasattr(torch.Tensor, 'mps') else None
t_nn_Module_mps = torch.nn.Module.mps if hasattr(torch.nn.Module, 'mps') else None

# Flag to prevent recursion
_patched = False


class DeviceType(type):
    """Metaclass for device type checking and construction."""
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, T_DEVICE_TYPE)
    
    def __call__(cls, *args: Any, **kwargs: Any) -> torch.device:
        return torch_device_replacement(*args, **kwargs)


class DeviceWrapper(metaclass=DeviceType):
    """Wrapper class for torch.device with custom type checking."""
    pass


@auto_log()
def get_default_device() -> torch.device:
    """Return the current default device."""
    global _default_device, _default_device_type
    if _default_device is None:
        with _device_lock:
            if _default_device is None:
                _detect_default_device_type()
                _default_device = t_device_constructor(_default_device_type)
                log_info(f"Initialized default device: {_default_device}")
    return _default_device


@auto_log()
def set_default_device(device: torch.device) -> None:
    """Set the default device."""
    global _default_device, _default_device_type
    with _device_lock:
        _default_device = device
        _default_device_type = device.type
        log_info(f"Set default device to: {device}")


@auto_log()
def _detect_default_device_type() -> str:
    """Detect the default device type based on available hardware."""
    global _default_device_type
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        _default_device_type = 'mps'
    elif torch.cuda.is_available() or torch.backends.cuda.is_built():
        _default_device_type = 'cuda'
    else:
        _default_device_type = 'cpu'
    return _default_device_type

@auto_log()
def cpu_override():
    return _cpu_override


@auto_log()
def torch_device_replacement(*args: Any, **kwargs: Any) -> torch.device:
    """
    Drop-in replacement for torch.device() with device redirection and CPU override toggle.
    • No arguments → returns default device (or CPU if override is active).
    • 'cpu:-1' or torch.device('cpu', -1) for override.
    • Redirects non-CPU devices to available hardware.
    • Preserves extra args and kwargs.
    Always returns a real torch.device instance after applying our redirection policy.
    """
    global _cpu_override, _default_device_type, _previous_default_device_type

    device_type = ""
    device_index = None
    with _device_lock:
        # If first argument is torch.device, check for override
        if args and isinstance(args[0], T_DEVICE_TYPE):
            return args[0]

        # If first argument is string device spec, parse and modify
        if args and isinstance(args[0], str):
            device_spec = args[0].strip()
            if ":" in device_spec:
                parts = device_spec.split(":", 1)
                device_type = parts[0].lower()
                try:
                    device_index = int(parts[1])
                except ValueError:
                    device_index = None
            else:
                device_type = device_spec.lower()

            # CPU override toggle logic
            if device_type == "cpu":
                if device_index == -1:
                    device_index = None
                    if cpu_override():
                        # Toggle OFF
                        _cpu_override = False
                        _default_device_type = _previous_default_device_type
                        _previous_default_device_type = ""
                    else:
                        # Toggle ON
                        _cpu_override = True
                        _previous_default_device_type = _default_device_type
                        _default_device_type = 'cpu'

        device_type = _default_device_type
        result = t_device_constructor(device_type, device_index)
        return result


@auto_log()
def tensor_to_replacement(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Replacement for torch.Tensor.to() that handles device conversion."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor_to_replacement called on non-tensor object: {type(tensor)}")
    
    # Handle device argument
    device = None
    if args and isinstance(args[0], (str, T_DEVICE_TYPE)):
        # Always redirect through the TorchDevice policy
        device = torch_device_replacement(args[0])
        args = (device,) + args[1:]
        kwargs.pop('device', None)
    elif 'device' in kwargs:
        # Always redirect through the TorchDevice policy
        device = torch_device_replacement(kwargs['device'])
        kwargs['device'] = device
    
    return t_Tensor_to(tensor, *args, **kwargs)


@auto_log()
def module_to_replacement(module: torch.nn.Module, *args, **kwargs) -> torch.nn.Module:
    """Replacement for torch.nn.Module.to() that handles device conversion."""
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"module_to_replacement called on non-module object: {type(module)}")
    
    # Handle device argument
    device = None
    if args and isinstance(args[0], (str, T_DEVICE_TYPE)):
        # Always redirect through the TorchDevice policy
        device = torch_device_replacement(args[0])
        args = (device,) + args[1:]
        kwargs.pop('device', None)
    elif 'device' in kwargs:
        # Always redirect through the TorchDevice policy
        device = torch_device_replacement(kwargs['device'])
        kwargs['device'] = device
    
    return t_nn_Module_to(module, *args, **kwargs)


@auto_log()
def tensor_cpu_replacement(tensor: torch.Tensor) -> torch.Tensor:
    """Replacement for torch.Tensor.cpu() that follows device redirection policy."""
    return tensor_to_replacement(tensor, torch_device_replacement('cpu'))


@auto_log()
def module_cpu_replacement(module: torch.nn.Module) -> torch.nn.Module:
    """Replacement for torch.nn.Module.cpu() that follows device redirection policy."""
    return module_to_replacement(module, torch_device_replacement('cpu'))


@auto_log()
def tensor_cuda_replacement(tensor: torch.Tensor, device: Optional[Union[int, torch.device]] = None) -> torch.Tensor:
    """Replacement for torch.Tensor.cuda() that follows device redirection policy."""
    return tensor_to_replacement(tensor, torch_device_replacement('cuda'))


@auto_log()
def module_cuda_replacement(module: torch.nn.Module, device: Optional[Union[int, torch.device]] = None) -> torch.nn.Module:
    """Replacement for torch.nn.Module.cuda() that follows device redirection policy."""
    return module_to_replacement(module, torch_device_replacement('cuda'))


@auto_log()
def tensor_mps_replacement(tensor: torch.Tensor) -> torch.Tensor:
    """Replacement for torch.Tensor.mps() that follows device redirection policy."""
    return tensor_to_replacement(tensor, torch_device_replacement('mps'))


@auto_log()
def module_mps_replacement(module: torch.nn.Module) -> torch.nn.Module:
    """Replacement for torch.nn.Module.mps() that follows device redirection policy."""
    return module_to_replacement(module, torch_device_replacement('mps'))


def apply_patches() -> None:
    """Apply all device-related patches."""
    global _patched
    if _patched:
        return
    
    # Set flag before patching to avoid recursion
    _patched = True
    
    # Ensure device detection happens first
    _detect_default_device_type()
    
    # Replace device constructor
    torch.device = DeviceWrapper
    
    # Add get_default_device to torch namespace
    torch.get_default_device = get_default_device
    
    # Patch tensor methods
    torch.Tensor.to = tensor_to_replacement
    torch.Tensor.cpu = tensor_cpu_replacement
    torch.Tensor.cuda = tensor_cuda_replacement
    if hasattr(torch.Tensor, 'mps'):
        torch.Tensor.mps = tensor_mps_replacement
    
    # Patch module methods
    torch.nn.Module.to = module_to_replacement
    torch.nn.Module.cpu = module_cpu_replacement
    torch.nn.Module.cuda = module_cuda_replacement
    if hasattr(torch.nn.Module, 'mps'):
        torch.nn.Module.mps = module_mps_replacement


__all__: List[str] = [
    'get_default_device',
    'set_default_device',
    'DeviceWrapper',
    'torch_device_replacement',
    'apply_patches'
]
