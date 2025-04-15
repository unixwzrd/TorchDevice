"""
TorchDevice library for managing PyTorch device operations.
This module patches PyTorch's CUDA functionality to work seamlessly with MPS (and vice-versa)
upon import.
"""

__version__ = '0.1.0'

from .TorchDevice import TorchDevice
from .modules.TDLogger import auto_log

# Automatically initialize TorchDevice to apply all patches.
TorchDevice.apply_patches()

__all__ = ['TorchDevice', 'auto_log', '__version__']