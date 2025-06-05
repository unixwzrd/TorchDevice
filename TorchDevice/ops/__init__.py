"""
TorchDevice Operations Module
---------------------------
Operation-specific patches and implementations.
"""

from TorchDevice.core.logger import log_info
from . import (
    device,
    memory,
    nn,
    random,
    streams,
    events,
    autograd
)

log_info("Initializing TorchDevice ops module")

__all__: list[str] = [
    'device',
    'memory',
    'nn',
    'random',
    'streams',
    'events',
    'autograd'
]

log_info("TorchDevice ops module initialized") 