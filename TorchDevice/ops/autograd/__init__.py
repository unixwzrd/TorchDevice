"""
TorchDevice Autograd Operations Module
--------------------------------
Automatic differentiation operations.
"""

from ...core.logger import log_info
from . import function, variable, grad_mode

log_info("Initializing TorchDevice autograd module")


def apply_patches() -> None:
    """Apply all autograd operation patches."""
    log_info("Applying autograd operation patches")
    
    # Apply patches from each submodule
    function.apply_patches()
    variable.apply_patches()
    grad_mode.apply_patches()
    
    log_info("Autograd operation patches applied")


__all__: list[str] = [
    'function',
    'variable',
    'grad_mode',
    'apply_patches'
]

log_info("TorchDevice autograd module initialized") 