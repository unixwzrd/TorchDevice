"""
TorchDevice Tensor Module
-------------------
Tensor operations and creation.
"""

from . import creation

print("Importing TorchDevice/ops/tensor/__init__.py")

__all__: list[str] = [
    'creation',
    'apply_patches'
]


def apply_patches() -> None:
    """Apply tensor-related patches."""
    print("Applying tensor patches")
    creation.apply_patches() 