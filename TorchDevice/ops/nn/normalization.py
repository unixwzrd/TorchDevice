"""
TorchDevice Neural Network Normalization Module
------------------------------------------
Normalization layer implementations.
"""

from TorchDevice.core.logger import log_info

log_info("Initializing TorchDevice nn normalization module")

__all__: list[str] = []  # Will be populated as we add functionality


def apply_patches() -> None:
    """Apply normalization patches."""
    log_info("Applying normalization patches")
    # No specific normalization layer patches are currently needed.
    # Device handling for normalization layers (subclasses of nn.Module)
    # is covered by the patches applied to torch.nn.Module methods (e.g., .to(), .cuda())
    # in ops.nn.layers.py and global tensor/device management.
    log_info("No specific normalization patches applied; covered by general nn.Module patches.")


log_info("TorchDevice nn normalization module initialized") 