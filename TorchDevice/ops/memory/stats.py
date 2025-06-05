"""
TorchDevice Memory Stats Module
---------------------------
Memory usage tracking and statistics.
"""

import torch
from typing import Optional
from TorchDevice.core.logger import log_info, auto_log

# Store original functions
t_cuda_memory_allocated = torch.cuda.memory_allocated if hasattr(torch.cuda, 'memory_allocated') else None
t_cuda_memory_reserved = torch.cuda.memory_reserved if hasattr(torch.cuda, 'memory_reserved') else None
t_cuda_max_memory_allocated = torch.cuda.max_memory_allocated if hasattr(torch.cuda, 'max_memory_allocated') else None
t_cuda_max_memory_reserved = torch.cuda.max_memory_reserved if hasattr(torch.cuda, 'max_memory_reserved') else None


@auto_log()
def memory_allocated(device: Optional[torch.device] = None) -> int:
    """Return the current GPU memory occupied by tensors in bytes."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_memory_allocated:
        return t_cuda_memory_allocated(device)
    return 0


@auto_log()
def memory_reserved(device: Optional[torch.device] = None) -> int:
    """Return the current GPU memory managed by the caching allocator in bytes."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_memory_reserved:
        return t_cuda_memory_reserved(device)
    return 0


@auto_log()
def max_memory_allocated(device: Optional[torch.device] = None) -> int:
    """Return the maximum GPU memory occupied by tensors in bytes."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_max_memory_allocated:
        return t_cuda_max_memory_allocated(device)
    return 0


@auto_log()
def max_memory_reserved(device: Optional[torch.device] = None) -> int:
    """Return the maximum GPU memory managed by the caching allocator in bytes."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_max_memory_reserved:
        return t_cuda_max_memory_reserved(device)
    return 0


def apply_patches() -> None:
    """Apply memory stats patches."""
    log_info("Applying memory stats patches")

    # Patch memory stats functions for CUDA
    if hasattr(torch.cuda, 'memory_allocated'):
        torch.cuda.memory_allocated = memory_allocated
    if hasattr(torch.cuda, 'memory_reserved'):
        torch.cuda.memory_reserved = memory_reserved
    if hasattr(torch.cuda, 'max_memory_allocated'):
        torch.cuda.max_memory_allocated = max_memory_allocated
    if hasattr(torch.cuda, 'max_memory_reserved'):
        torch.cuda.max_memory_reserved = max_memory_reserved

    log_info("Memory stats patches applied")


# Module initialization
log_info("Initializing TorchDevice memory stats module")

__all__: list[str] = [
    'memory_allocated',
    'memory_reserved',
    'max_memory_allocated',
    'max_memory_reserved',
    'apply_patches'
]

log_info("TorchDevice memory stats module initialized")
