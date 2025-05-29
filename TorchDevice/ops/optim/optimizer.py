"""
TorchDevice Optimizer Module
-----------------------
Optimizer implementations.
"""

from typing import List
from ...core.logger import log_info

log_info("Importing TorchDevice/ops/optim/optimizer.py")


def apply_patches() -> None:
    """Apply optimizer patches."""
    log_info("Applying optimizer patches")


__all__: List[str] = [
    'apply_patches'
]
