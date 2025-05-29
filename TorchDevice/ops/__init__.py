"""
TorchDevice Operations Module
-------------------------
Contains all device-specific operations.
"""

from . import (
    autograd,
    events,
    memory,
    nn,
    optim,
    random,
    streams,
    tensor
)

print("Importing TorchDevice/ops/__init__.py")

__all__: list[str] = [
    'autograd',
    'events',
    'memory',
    'nn',
    'optim',
    'random',
    'streams',
    'tensor',
    'apply_patches'
]


def apply_patches() -> None:
    """Apply all operation-related patches."""
    print("Applying operation patches")
    autograd.apply_patches()
    events.apply_patches()
    memory.apply_patches()
    nn.apply_patches()
    optim.apply_patches()
    random.apply_patches()
    streams.apply_patches()
    tensor.apply_patches()
