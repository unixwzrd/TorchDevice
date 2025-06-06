"""
TorchDevice Device Modules
-------------------------
This package contains device-specific patches and functionality.
"""

from . import memory
from . import random
from . import streams
from . import unassigned
from . import nn

__all__ = ['memory', 'random', 'streams', 'unassigned', 'nn']
