"""
TorchDevice Random Operations Module
--------------------------------
Random number generation and seed management.
"""

from ...core.logger import log_info
from . import generators, distributions

log_info("Initializing TorchDevice random operations module")

def apply_patches() -> None:
    """Apply all random operation patches."""
    log_info("Applying random operation patches")
    generators.apply_patches()
    distributions.apply_patches()
    log_info("Random operation patches applied")

__all__: list[str] = [
    'generators',
    'distributions',
    'apply_patches'
]

log_info("TorchDevice random operations module initialized") 