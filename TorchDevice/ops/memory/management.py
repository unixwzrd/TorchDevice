"""
TorchDevice Memory Management Module
--------------------------------
Memory allocation and cache management.
"""

import torch
from TorchDevice.core.logger import log_info, auto_log

# Store original functions
t_cuda_empty_cache = torch.cuda.empty_cache if hasattr(torch.cuda, 'empty_cache') else None
t_mps_empty_cache = torch.mps.empty_cache if hasattr(torch.mps, 'empty_cache') else None


@auto_log()
def empty_cache() -> None:
    """Empty the memory cache for the current device."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device_type = DeviceManager.get_default_device().type
    if device_type == 'cuda' and t_cuda_empty_cache:
        t_cuda_empty_cache()
    elif device_type == 'mps' and t_mps_empty_cache:
        t_mps_empty_cache()


def apply_patches() -> None:
    """Apply memory management patches."""
    log_info("Applying memory management patches")
    
    # Patch empty_cache function for both CUDA and MPS
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache = empty_cache
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache = empty_cache
    
    log_info("Memory management patches applied")


# Module initialization
log_info("Initializing TorchDevice memory management module")

__all__: list[str] = [
    'empty_cache',
    'apply_patches'
]

log_info("TorchDevice memory management module initialized") 