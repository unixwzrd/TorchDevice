"""
TorchDevice library for managing PyTorch device operations.
This module patches PyTorch's CUDA functionality to work seamlessly with MPS (and vice-versa)
upon import.
"""

__version__ = '0.0.6'

from .TorchDevice import TorchDevice, initialize_torchdevice
from .modules.TDLogger import auto_log

# Automatically initialize TorchDevice to apply all patches.
initialize_torchdevice()

__all__ = ['TorchDevice', 'initialize_torchdevice', 'auto_log', '__version__']