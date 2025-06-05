"""
TorchDevice Core Module
----------------------
Core functionality for device handling, patching, and logging.
"""

from .logger import log_info
from . import logger, patch, tensors, device, hardware_info

log_info("Initializing TorchDevice core module")

__all__: list[str] = [
    'device',
    'tensors',
    'logger',
    'patch',
    'hardware_info'
]

log_info("TorchDevice core module initialized")