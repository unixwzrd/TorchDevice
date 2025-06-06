"""
TorchDevice Core Module
----------------------
Core functionality for device handling, patching, and logging.
"""

from .logger import log_info
from . import logger, patch, tensors, device, hardware_info

log_info("Initializing TorchDevice core module")

def apply_patches() -> None:
    """Apply all core module patches."""
    log_info("Applying core module patches")
    tensors.apply_patches()
    # Other core modules like logger, patch, device, hardware_info
    # do not have an apply_patches() in the same vein as they provide
    # foundational utilities or manage state/patching mechanisms themselves.
    log_info("Core module patches applied")

__all__: list[str] = [
    'device',
    'tensors',
    'logger',
    'patch',
    'hardware_info',
    'apply_patches' # Added apply_patches
]

log_info("TorchDevice core module initialized")