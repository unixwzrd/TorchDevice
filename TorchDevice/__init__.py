"""
TorchDevice library for managing PyTorch device operations.
"""

__version__ = '0.0.2'

from .TorchDevice import TorchDevice
from .modules.TDLogger import log_message

# Create a singleton instance
torch_device = TorchDevice()

__all__ = ['TorchDevice', 'torch_device', 'log_message', '__version__']