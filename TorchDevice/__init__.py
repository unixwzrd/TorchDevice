"""
TorchDevice library for managing PyTorch device operations.
This module patches PyTorch's CUDA functionality to work seamlessly with MPS (and vice-versa)
upon import. Users should never need to call patch functions directly—patching is automatic.
"""

__version__ = '0.2.0'

from .TorchDevice import TorchDevice
from .modules.TDLogger import auto_log
from .modules import patch
from .modules import compile

# Apply all monkey-patches automatically on import
# Users should never call patch functions directly.
TorchDevice.get_default_device()
patch.apply_all_patches()


# Expose a function to apply deferred patches - these must be run after core patching
def apply_deferred_patches():
    """Apply patches that must be run after the core system is initialized."""
    compile.patch_dynamo_config()


# Run deferred patches - but only after import is complete
# This prevents circular import issues
from . import _deferred_patches  # noqa

__all__ = ['TorchDevice', 'auto_log', '__version__']
