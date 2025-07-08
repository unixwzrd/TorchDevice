"""
TorchDevice Streams Operations Module
--------------------------------
Stream and event management for device operations.
"""

from ...core.logger import log_info
from . import cuda, mps, sync

log_info("Initializing TorchDevice streams module")


def apply_patches() -> None:
    """Apply all stream operation patches."""
    log_info("Applying stream operation patches")
    
    # Apply patches from each submodule
    cuda.apply_patches()
    mps.apply_patches()
    sync.apply_patches()
    
    log_info("Stream operation patches applied")


__all__: list[str] = [
    'cuda',
    'mps',
    'sync',
    'apply_patches'
]

log_info("TorchDevice streams module initialized") 