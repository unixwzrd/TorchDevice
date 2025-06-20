"""
TorchDevice Neural Network Operations Module
---------------------------------------
Neural network operations and layers.
"""

from TorchDevice.core.logger import log_info
from . import (
    containers,
    layers,
    normalization,
    activation,
    attention,
    init
)

log_info("Initializing TorchDevice nn module")


def apply_patches() -> None:
    """Apply all neural network patches."""
    log_info("Applying neural network patches")
    
    # Apply patches from each submodule
    containers.apply_patches()
    layers.apply_patches()
    normalization.apply_patches()
    activation.apply_patches()
    attention.apply_patches()
    init.apply_patches()
    
    log_info("Neural network patches applied")


__all__: list[str] = [
    'containers',
    'layers',
    'normalization',
    'activation',
    'attention',
    'init',
    'apply_patches'
]

log_info("TorchDevice nn module initialized") 