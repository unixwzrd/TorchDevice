"""
TorchDevice Stream Synchronization Module
-----------------------------------
Stream synchronization and event management.
"""

import torch
from typing import Optional
from TorchDevice.core.logger import log_info, auto_log

# Store original functions
t_cuda_synchronize = torch.cuda.synchronize if hasattr(torch.cuda, 'synchronize') else None
t_mps_synchronize = torch.mps.synchronize if hasattr(torch.mps, 'synchronize') else None


@auto_log()
def synchronize(device: Optional[torch.device] = None) -> None:
    """Synchronize the current device."""
    from TorchDevice.core.device import DeviceManager  # Local import
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
        torch.cuda.synchronize = synchronize
    if hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize = synchronize
    
    log_info("Stream synchronization patches applied")

# Module initialization
log_info("Initializing TorchDevice stream synchronization module")

__all__: list[str] = [
    'synchronize',
    'apply_patches'
]

log_info("TorchDevice stream synchronization module initialized")