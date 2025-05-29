"""
TorchDevice Autograd Grad Mode Module
------------------------------
Gradient computation mode control.
"""

from typing import List
from ...core.logger import log_info

log_info("Importing TorchDevice/ops/autograd/grad_mode.py")


def apply_patches() -> None:
    """Apply grad mode patches."""
    log_info("Applying grad mode patches")


__all__: List[str] = [
    'apply_patches'
]
