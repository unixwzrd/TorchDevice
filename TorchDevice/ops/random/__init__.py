"""
TorchDevice Random Module
--------------------
Random number generation functionality.
"""

from . import (
    generators,
    distributions
)

print("Importing TorchDevice/ops/random/__init__.py")

__all__: list[str] = [
    'generators',
    'distributions',
    'apply_patches'
]

def apply_patches() -> None:
    """Apply random number generation patches."""
    print("Applying random number generation patches")
    generators.apply_patches()
    distributions.apply_patches()
