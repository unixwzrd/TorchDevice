"""
TorchDevice Autograd Variable Module
-----------------------------
Variable handling and tracking.
"""

from typing import List
from ...core.logger import log_info

log_info("Importing TorchDevice/ops/autograd/variable.py")


def apply_patches() -> None:
    """Apply autograd variable patches."""
    log_info("Applying autograd variable patches")


__all__: List[str] = [
    'apply_patches'
]
