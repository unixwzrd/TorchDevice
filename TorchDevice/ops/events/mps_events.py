"""
TorchDevice MPS Events Module
-----------------------
MPS event implementations.
"""

from typing import List
from ...core.logger import log_info

log_info("Importing TorchDevice/ops/events/mps_events.py")


def apply_patches() -> None:
    """Apply MPS event patches."""
    log_info("Applying MPS event patches")


__all__: List[str] = [
    'apply_patches'
]
