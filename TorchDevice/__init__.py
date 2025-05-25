"""
TorchDevice library for managing PyTorch device operations.
This module patches PyTorch's CUDA functionality to work seamlessly with MPS (and vice-versa)
upon import. Users should never need to call patch functions directly—patching is automatic.
"""

__version__ = '0.4.0'

from .TorchDevice import TorchDevice
from .modules.TDLogger import auto_log
from .device import nn, attention

# Apply all monkey-patches automatically on import
# Users should never call patch functions directly.
TorchDevice.apply_patches()

# Expose key functions at module level
get_default_device = TorchDevice.get_default_device

__all__ = ['TorchDevice', 'auto_log', '__version__', 'nn', 'attention', 'get_default_device']
