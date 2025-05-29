"""
TorchDevice Device Context Module
-----------------------------
Device context management and emulation.
"""

import torch
import contextlib
from typing import Any, Optional
from ...core.logger import auto_log

@auto_log()
def _device_context(device: Any = 0):
    """Device context manager that masquerades as CUDA context manager."""
    yield

@auto_log()
def _set_device(device: Optional[Any]) -> None:
    """Set the current device (no-op for non-CUDA devices)."""
    pass

@auto_log()
def _current_device() -> int:
    """Get the current device index (always returns 0 for non-CUDA devices)."""
    return 0

@auto_log()
def _device_count() -> int:
    """Get the number of available devices (always returns 1 for non-CUDA devices)."""
    return 1

def apply_patches() -> None:
    """Apply device context-related patches to torch.cuda."""
    # Patch CUDA context manager
    torch.cuda.device = contextlib.contextmanager(_device_context)
    torch.cuda.set_device = _set_device
    torch.cuda.current_device = _current_device
    torch.cuda.device_count = _device_count 