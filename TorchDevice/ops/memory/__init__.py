"""
TorchDevice Memory Operations Module
--------------------------------
Memory management and tracking.
"""

from TorchDevice.core.logger import log_info
from . import management, stats

log_info("Initializing TorchDevice memory module")


def apply_patches() -> None:
    """Apply all memory operation patches."""
    log_info("Applying memory operation patches")
    management.apply_patches()
    stats.apply_patches()
    log_info("Memory operation patches applied")


__all__: list[str] = [
    'management',
    'stats',
    'apply_patches'
]

log_info("TorchDevice memory module initialized") 