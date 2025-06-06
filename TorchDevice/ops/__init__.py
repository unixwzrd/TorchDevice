"""
TorchDevice Operations Module
---------------------------
Operation-specific patches and implementations.
"""

from TorchDevice.core.logger import log_info
from . import (
    device,
    memory,
    nn,
    random,
    streams,
    events,
    autograd
)

log_info("Initializing TorchDevice ops module")

def apply_patches() -> None:
    """Apply all operation-specific patches."""
    log_info("Applying all ops module patches")
    device.apply_patches()
    memory.apply_patches()
    nn.apply_patches()
    random.apply_patches()
    streams.apply_patches()
    events.apply_patches()
    autograd.apply_patches()
    log_info("All ops module patches applied")

__all__: list[str] = [
    'device',
    'memory',
    'nn',
    'random',
    'streams',
    'events',
    'autograd',
    'apply_patches' # Added apply_patches
]

log_info("TorchDevice ops module initialized") 