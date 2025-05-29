"""
TorchDevice Autograd Function Module
-----------------------------
Custom autograd function implementations.
"""

from typing import List
from ...core.logger import log_info

log_info("Importing TorchDevice/ops/autograd/function.py")


def apply_patches() -> None:
    """Apply autograd function patches."""
    log_info("Applying autograd function patches")


__all__: List[str] = [
    'apply_patches'
]
