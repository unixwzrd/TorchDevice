"""
TorchDevice Device Operations Module
--------------------------------
Device-specific operations and patches.
"""

from TorchDevice.core.logger import log_info
from . import cuda, mps, cpu

log_info("Initializing TorchDevice device operations module")

__all__: list[str] = [
    'cuda',
    'mps',
    'cpu'
]

log_info("TorchDevice device operations module initialized") 