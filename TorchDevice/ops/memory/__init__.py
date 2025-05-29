"""
TorchDevice Memory Module
---------------------
Memory management and monitoring.
"""

from . import management
from . import stats

print("Importing TorchDevice/ops/memory/__init__.py")

__all__: list[str] = [
    'management',
    'stats',
    'apply_patches'
]

def apply_patches() -> None:
    """Apply memory-related patches."""
    print("Applying memory patches")
    management.apply_patches()
    stats.apply_patches()
