"""
TorchDevice CUDA Events Module
--------------------------
CUDA event implementation for non-CUDA devices.
"""

import torch
import time
from typing import Optional, Any
from ...core.logger import auto_log, log_warning
from ...core.device import get_default_device, torch_device_replacement

# --- Base Class Resolution Helpers ---

def _get_event_base():
    """Get the base class for event implementation."""
    try:
        from torch._streambase import _EventBase
        return _EventBase
    except (AttributeError, ImportError):
        try:
            from torch._C import _EventBase
            return _EventBase
        except (AttributeError, ImportError):
            try:
                return torch._C._EventBase
            except (AttributeError, ImportError):
                log_warning(
                    "torch._streambase._EventBase not found, using object as base class for Event - "
                    "this may cause issues with PyTorch dynamo"
                )
                return object

_EventBase = _get_event_base()

# --- Event Replacement ---

class Event(_EventBase):
    """Replacement for torch.cuda.Event (MPS/CPU)."""

    @auto_log()
    def __init__(self, enable_timing: bool = False, blocking: bool = False, 
                 interprocess: bool = False, device: Optional[Any] = None):
        if _EventBase is not object:
            try:
                super().__init__()
            except Exception as e:
                log_warning(f"Error calling _EventBase.__init__: {e}")
        if device is None:
            device = get_default_device()
        self._device = torch_device_replacement(device)
        self.enable_timing = enable_timing
        self.blocking = blocking
        self.interprocess = interprocess
        self._is_created = True
        self._is_destroyed = False
        self._recorded = False
        self._record_time = None
        self._stream = None

    @auto_log()
    def record(self, stream=None):
        if stream is not None:
            stream_device = getattr(stream, 'device', None)
            if stream_device is not None and self._device != stream_device:
                log_warning(f"Event and stream device mismatch: {self._device} vs {stream_device}")
            self._stream = stream
        self._recorded = True
        self._record_time = time.time()
        return self

    @auto_log()
    def wait(self, stream=None):
        if not self._recorded:
            log_warning("Event has not been recorded yet")
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        return self

    @auto_log()
    def query(self):
        return self._recorded

    @auto_log()
    def elapsed_time(self, end_event):
        if not self.enable_timing:
            log_warning("Events were created without timing enabled, but returning mock value anyway")
            return 0.5
        if not self._recorded or not getattr(end_event, '_recorded', False):
            log_warning("One or both events have not been recorded, returning mock value")
            return 0.5
        start_time = self._record_time or 0
        end_time = getattr(end_event, '_record_time', time.time()) or 0
        elapsed_ms = (end_time - start_time) * 1000.0
        return elapsed_ms

    @auto_log()
    def synchronize(self):
        if not self._recorded:
            log_warning("Event has not been recorded yet")
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        return self

    @property
    def is_completed(self):
        return self._recorded

    @property
    def ipc_handle(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __del__(self):
        if hasattr(self, '_is_destroyed') and not self._is_destroyed:
            self._is_destroyed = True

    def __str__(self):
        return f"Event(device={self._device}, timing={self.enable_timing})"

    def __eq__(self, o):
        if isinstance(o, Event):
            return (self._device == o._device and self.enable_timing == o.enable_timing)
        return False

    def __hash__(self):
        return hash((self._device, self.enable_timing))

def apply_patches():
    """Apply all CUDA event-related patches."""
    torch.cuda.Event = Event
