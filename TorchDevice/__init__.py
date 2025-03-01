"""
TorchDevice: A package for seamless PyTorch device management
"""

# Import the main TorchDevice class
from .TorchDevice import TorchDevice

# Apply patches to make CUDA functions work on MPS
TorchDevice.apply_patches()

__all__ = ['TorchDevice']