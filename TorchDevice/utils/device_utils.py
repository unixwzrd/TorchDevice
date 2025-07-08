"""
TorchDevice Device Utilities Module
-------------------------------
Device management and utility functions.
"""

import torch
from ..core.device import DeviceManager
from ..core.logger import log_info, auto_log

log_info("Initializing TorchDevice device utils module")

@auto_log()
def is_cuda_effectively_available() -> bool:
    """Checks if CUDA is effectively available (either native CUDA or MPS fallback)."""
    # This relies on torch.cuda.is_available() being patched by TorchDevice
    # to return True if MPS is acting as a CUDA replacement.
    return torch.cuda.is_available()

@auto_log()
def is_mps_effectively_available() -> bool:
    """Checks if MPS is available and built."""
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()

@auto_log()
def get_current_device_type() -> str:
    """Gets the current primary device type TorchDevice is using ('cuda', 'mps', or 'cpu')."""
    # Ensure DeviceManager is initialized to get the correct default type
    DeviceManager.get_default_device() 
    return DeviceManager._default_device_type

@auto_log()
def is_cpu_override_active() -> bool:
    """Checks if the CPU override is currently active in TorchDevice."""
    return DeviceManager.cpu_override()


__all__: list[str] = [
    'is_cuda_effectively_available',
    'is_mps_effectively_available',
    'get_current_device_type',
    'is_cpu_override_active'
]

def apply_patches() -> None:
    """Apply device utility patches."""
    log_info("Applying device utility patches")
    # No patches are applied by this module. It provides utility functions only.
    log_info("No device utility patches needed.")


log_info("TorchDevice device utils module initialized")