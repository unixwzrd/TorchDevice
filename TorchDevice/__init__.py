"""
TorchDevice library for managing PyTorch device operations.
This module patches PyTorch's CUDA functionality to work seamlessly with MPS (and vice-versa)
upon import. Users should never need to call patch functions directly—patching is automatic.
"""

<<<<<<< HEAD
<<<<<<< HEAD
__version__ = '0.0.5'
=======
__version__ = '0.1.1'
>>>>>>> 20250427_00-rollback
=======
__version__ = '0.2.0'
>>>>>>> rollback-20250506

from .TorchDevice import TorchDevice
from .modules.TDLogger import auto_log

# Apply all monkey-patches automatically on import
# Users should never call patch functions directly.
TorchDevice.apply_patches()

__all__ = ['TorchDevice', 'auto_log', '__version__']
