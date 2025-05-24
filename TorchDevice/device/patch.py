"""
TorchDevice Patch Application
---------------------------
Centralizes all patch application logic for TorchDevice.
"""

from . import memory
from . import random
from . import streams
from . import unassigned
from . import nn

def apply_all_patches():
    """Apply all TorchDevice patches."""
    memory.apply_patches()
    random.apply_patches()
    streams.apply_patches()
    unassigned.apply_patches()
    nn.apply_patches() 