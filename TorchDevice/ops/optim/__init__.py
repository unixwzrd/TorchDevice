"""
TorchDevice Optimization Module
-------------------------
Optimization algorithms and learning rate scheduling.
"""

from . import (
    optimizer,
    lr_scheduler
)

print("Importing TorchDevice/ops/optim/__init__.py")

__all__: list[str] = [
    'optimizer',
    'lr_scheduler',
    'apply_patches'
]


def apply_patches() -> None:
    """Apply optimization-related patches."""
    print("Applying optimization patches")
    optimizer.apply_patches()
    lr_scheduler.apply_patches()

