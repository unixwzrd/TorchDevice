"""
TorchDevice Streams Module
---------------------
Stream handling and synchronization.
"""

from . import (
    cuda,
    mps,
    synchronize
)

print("Importing TorchDevice/ops/streams/__init__.py")

__all__: list[str] = [
    'cuda',
    'mps',
    'synchronize',
    'apply_patches'
]


def apply_patches() -> None:
    """Apply stream-related patches."""
    print("Applying stream patches")
    cuda.apply_patches()
    mps.apply_patches()
    synchronize.apply_patches()
