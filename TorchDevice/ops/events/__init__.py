"""
TorchDevice Events Module
-------------------
Event handling and synchronization.
"""

from . import (
    cuda_events,
    mps_events,
    synchronize
)

print("Importing TorchDevice/ops/events/__init__.py")

__all__: list[str] = [
    'cuda_events',
    'mps_events',
    'synchronize',
    'apply_patches'
]


def apply_patches() -> None:
    """Apply event-related patches."""
    print("Applying event patches")
    cuda_events.apply_patches()
    mps_events.apply_patches()
    synchronize.apply_patches()

