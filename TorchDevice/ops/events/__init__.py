"""
TorchDevice Events Operations Module
-------------------------------
Event management and synchronization.
"""

from ...core.logger import log_info
from . import cuda, mps, sync

log_info("Initializing TorchDevice events module")


def apply_patches() -> None:
    """Apply all event operation patches."""
    log_info("Applying event operation patches")
    
    # Apply patches from each submodule
    cuda.apply_patches()
    mps.apply_patches()
    sync.apply_patches()
    
    log_info("Event operation patches applied")


__all__: list[str] = [
    'cuda',
    'mps',
    'sync',
    'apply_patches'
]

log_info("TorchDevice events module initialized") 