"""
TorchDevice Autograd Module
----------------------
Automatic differentiation functionality.
"""

from . import (
    function,
    variable,
    grad_mode
)

print("Importing TorchDevice/ops/autograd/__init__.py")

__all__: list[str] = [
    'function',
    'variable',
    'grad_mode',
    'apply_patches'
]


def apply_patches() -> None:
    """Apply autograd-related patches."""
    print("Applying autograd patches")
    function.apply_patches()
    variable.apply_patches()
    grad_mode.apply_patches()

# Placeholder for autograd functionality

