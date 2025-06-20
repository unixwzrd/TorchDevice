"""
TorchDevice MPS Operations Module
-----------------------------
MPS-specific operations and patches.
"""

import torch
from TorchDevice.core.logger import log_info
# Tensor movement functions are now in core.tensors

# Tensor and module MPS replacement functions are now in core.tensors module



def apply_patches() -> None:
    """Apply MPS-specific patches."""
    log_info("Applying MPS patches")

    # Tensor movement functions are now applied from core.tensors module
    # No MPS-specific patches needed at this time

    log_info("MPS patches applied")


# Module initialization
log_info("Initializing TorchDevice MPS operations module")


__all__: list[str] = [
    'apply_patches',
    '_is_mps_actually_available'
]

log_info("TorchDevice MPS operations module initialized")


def _is_mps_actually_available() -> bool:
    """Check if MPS is actually available and built."""
    # Check both is_available() and is_built() for robustness, similar to CUDA checks
    # and how DeviceManager._detect_default_device_type handles it.
    available = torch.backends.mps.is_available()
    built = torch.backends.mps.is_built()
    log_info(f"MPS available: {available}, MPS built: {built}")
    return available and built
