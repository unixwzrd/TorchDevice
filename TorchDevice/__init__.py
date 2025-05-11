"""
TorchDevice library for managing PyTorch device operations.
This module patches PyTorch's CUDA functionality to work seamlessly with MPS (and vice-versa)
upon import. Users should never need to call patch functions directly—patching is automatic.
"""

__version__ = '0.1.1'

from .TorchDevice import TorchDevice
from .modules.TDLogger import auto_log
from .cuda import patch

# Apply all monkey-patches automatically on import
# Users should never call patch functions directly.
patch.apply_all_patches()
TorchDevice.get_default_device()

__all__ = ['TorchDevice', 'auto_log', '__version__']
