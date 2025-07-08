"""
TorchDevice Neural Network Activation Module
---------------------------------------
Activation function implementations.
"""

from ...core.logger import log_info

log_info("Initializing TorchDevice nn activation module")

__all__: list[str] = []  # Will be populated as we add functionality


def apply_patches() -> None:
    """Apply activation patches."""
    log_info("Applying activation patches")
    # No specific activation function module patches are currently needed.
    # Device handling for activation layers (subclasses of nn.Module)
    # is covered by the patches applied to torch.nn.Module methods (e.g., .to(), .cuda())
    # in ops.nn.layers.py and global tensor/device management.
    # Functional activations (e.g., torch.nn.functional.relu) operate on tensors whose
    # device affinity is already managed by TorchDevice.
    log_info("No specific activation patches applied; covered by general nn.Module and tensor patches.")


log_info("TorchDevice nn activation module initialized") 