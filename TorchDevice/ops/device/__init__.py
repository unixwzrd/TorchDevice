"""
TorchDevice Device Operations Module
--------------------------------
Device-specific operations and patches.
"""

from TorchDevice.core.logger import log_info
from . import cuda, mps, cpu

log_info("Initializing TorchDevice device operations module")

def apply_patches() -> None:
    """Apply all device operation patches."""
    log_info("Applying device operation patches")
    cuda.apply_patches()
    mps.apply_patches()
    cpu.apply_patches()
    log_info("Device operation patches applied")

__all__: list[str] = [
    'cuda',
    'mps',
    'cpu',
    'apply_patches' # Added apply_patches
]

log_info("TorchDevice device operations module initialized")
