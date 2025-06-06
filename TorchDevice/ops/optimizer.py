"""
TorchDevice Optimizer Functions Module
------------------------------------
Handles any necessary patches or wrappers for PyTorch optimizers.
Currently, no specific patches are applied as standard PyTorch optimizers
should work correctly if model parameters are on the correct device.
"""

import torch
from TorchDevice.core.logger import log_info, auto_log

_patches_applied = False

@auto_log()
def apply_patches() -> None:
    """Apply optimizer related patches. Currently none are needed."""
    global _patches_applied
    if _patches_applied:
        return

    log_info("Applying Optimizer patches")
    # No specific optimizer patches are currently implemented in TorchDevice.
    # Standard PyTorch optimizers are expected to work correctly provided that
    # model parameters and their gradients are on the appropriate device, which
    # is handled by other TorchDevice patches (tensor.to, module.to, etc.).
    log_info("No specific optimizer patches needed at this time.")
    _patches_applied = True
    log_info("Optimizer patches (none) applied")

__all__ = ['apply_patches']

log_info("TorchDevice Optimizer module initialized")
