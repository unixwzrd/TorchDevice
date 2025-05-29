"""
TorchDevice Neural Network Module
-------------------------
Neural network operations.
"""

from typing import List
from ...core.logger import log_info

log_info("Importing TorchDevice/ops/nn/__init__.py")


def apply_patches() -> None:
    """Apply neural network patches."""
    log_info("Applying neural network patches")


__all__: List[str] = [
    'apply_patches'
]
