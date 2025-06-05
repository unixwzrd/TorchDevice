"""
TorchDevice - Transparent PyTorch Device Redirection
-------------------------------------------------
This module enables seamless code portability between NVIDIA CUDA, Apple Silicon (MPS),
and CPU hardware for PyTorch applications.
"""

__version__ = '0.2.0'

from .core.logger import log_info, auto_log
from .core import patch

log_info("Initializing TorchDevice package")

# Apply all patches when the module is imported
patch.ensure_patched()

__all__ = ['__version__', 'auto_log']

log_info("TorchDevice package initialized") 