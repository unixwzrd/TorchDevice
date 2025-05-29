"""
TorchDevice Core Module
--------------------
Core functionality for device handling and redirection.
"""

from . import device
from . import patch
from . import logger

print("Importing TorchDevice/core/__init__.py")

__all__: list[str] = [
    'device',
    'patch',
    'logger'
]

