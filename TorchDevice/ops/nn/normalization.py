"""
TorchDevice Neural Network Normalization Module
----------------------------------------
Normalization layer implementations.
"""

from typing import List
from ...core.logger import log_info

log_info("Importing TorchDevice/ops/nn/normalization.py")

def apply_patches() -> None:
    """Apply normalization patches."""
    log_info("Applying normalization patches")

__all__: List[str] = [
    'apply_patches'
]
