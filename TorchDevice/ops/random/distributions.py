"""
TorchDevice Random Distributions Module
--------------------------------
Probability distribution implementations.
"""

from typing import List
from ...core.logger import log_info

log_info("Importing TorchDevice/ops/random/distributions.py")


def apply_patches() -> None:
    """Apply random distributions patches."""
    log_info("Applying random distributions patches")


__all__: List[str] = [
    'apply_patches'
]
