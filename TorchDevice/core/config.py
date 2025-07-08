"""
TorchDevice Core Configuration Module
-----------------------------------
This module holds global configuration and state for TorchDevice, such as
exclusion lists and context managers for controlling patching behavior.
"""

import threading
from contextlib import contextmanager
from typing import Set, ContextManager

# --- Thread-Local State for Bypass --- #
_bypass_state = threading.local()


def _init_bypass_state() -> None:
    """Initializes the thread-local state for bypassing argument processing."""
    if not hasattr(_bypass_state, 'active'):
        _bypass_state.active = False


_init_bypass_state()  # Initialize on module load


# --- Public Functions and Context Managers --- #
def is_bypass_active() -> bool:
    """Checks if the argument processing bypass is currently active."""
    _init_bypass_state() # Ensure it's initialized for the current thread
    return _bypass_state.active


@contextmanager
def bypass_argument_processing() -> ContextManager[None]:
    """A context manager to temporarily disable tensor device redirection."""
    _init_bypass_state()
    original_state = _bypass_state.active
    _bypass_state.active = True
    try:
        yield
    finally:
        _bypass_state.active = original_state


# --- Exclusion Lists --- #
# Functions whose arguments should not be processed or moved to the target device.
# This is for functions that have specific CPU-only tensor requirements.
ARGUMENT_PROCESSING_EXCLUSIONS: Set[str] = {
    'torch.nn.utils.rnn.pack_padded_sequence',
}

__all__ = [
    'is_bypass_active',
    'bypass_argument_processing',
    'ARGUMENT_PROCESSING_EXCLUSIONS',
]
