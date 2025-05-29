"""
TorchDevice MPS Streams Module
------------------------
MPS stream implementations.
"""

from typing import List
from ...core.logger import log_info

log_info("Importing TorchDevice/ops/streams/mps.py")


class MPSEvent:
    """MPS event class."""
    def __init__(self) -> None:
        pass

    def synchronize(self) -> None:
        """Synchronize the event."""
        pass


def apply_patches() -> None:
    """Apply MPS stream patches."""
    log_info("Applying MPS stream patches")


__all__: List[str] = [
    'MPSEvent',
    'apply_patches'
]
