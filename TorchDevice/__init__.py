"""
TorchDevice - A PyTorch Device Abstraction Layer
----------------------------------------------
Enables seamless code portability between NVIDIA CUDA, Apple Silicon (MPS),
and CPU hardware for PyTorch applications.
"""

__version__ = "0.1.0"

from .core.device import get_default_device, set_default_device
from .core.patch import ensure_patches_applied

# Apply patches when the module is imported
ensure_patches_applied()

__all__ = [
    'get_default_device',
    'set_default_device',
]


