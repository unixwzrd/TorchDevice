"""
TorchDevice Neural Network Initialization Module
----------------------------------------
Parameter initialization utilities.
"""

from typing import List
from ...core.logger import log_info

log_info("Importing TorchDevice/ops/nn/init.py")


def apply_patches() -> None:
    """Apply initialization patches."""
    log_info("Applying initialization patches")


__all__: List[str] = [
    'apply_patches'
]
