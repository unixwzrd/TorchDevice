"""
TorchDevice Stream Synchronization Module
-----------------------------------
Stream synchronization and event management.
"""

import torch
from typing import Optional, Any
from ...core.logger import log_info, auto_log
from ...core.device import DeviceManager

# Store original functions
t_cuda_synchronize = torch.cuda.synchronize if hasattr(torch.cuda, 'synchronize') else None
t_mps_synchronize = torch.mps.synchronize if hasattr(torch.mps, 'synchronize') else None


@auto_log()
def t_cuda_synchronize_function(device: Optional[Any] = None) -> None:
    """Replacement for torch.cuda.synchronize, handles MPS/CPU redirection."""
    device_type = torch.device(DeviceManager.torch_device_replacement(device)).type if device is not None else DeviceManager.get_default_device().type
    if device_type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize()
    # For CPU or other types (including actual CUDA if not redirected), this is a no-op,
    # as the original function would have been called if it were a true CUDA context,
    # or PyTorch handles the no-op for CPU.


@auto_log()
def synchronize(device: Optional[torch.device] = None) -> None:
    """Synchronize the current device."""
    from ...core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_synchronize:
        t_cuda_synchronize(device)
    elif device.type == 'mps' and t_mps_synchronize:
        t_mps_synchronize()


def apply_patches() -> None:
    """Apply stream synchronization patches."""
    log_info("Applying stream synchronization patches")
    
    # Patch synchronize functions
    if hasattr(torch.cuda, 'synchronize'):
        torch.cuda.synchronize = t_cuda_synchronize_function # Use the new function for CUDA sync
    if hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize = synchronize
    
    log_info("Stream synchronization patches applied")

# Module initialization
log_info("Initializing TorchDevice stream synchronization module")

__all__: list[str] = [
    'synchronize',
    't_cuda_synchronize_function',
    'apply_patches'
]

log_info("TorchDevice stream synchronization module initialized")
