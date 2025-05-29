"""
TorchDevice CUDA Stream Module
--------------------------
CUDA stream implementation for non-CUDA devices.
"""

import torch
from typing import Optional, Any
from ...core.logger import auto_log, log_warning
from ...core.device import get_default_device, torch_device_replacement

# --- Base Class Resolution Helpers ---

def _get_stream_base():
    """Get the base class for stream implementation."""
    try:
        from torch._streambase import _StreamBase
        return _StreamBase
    except (AttributeError, ImportError):
        try:
            from torch._C import _StreamBase
            return _StreamBase
        except (AttributeError, ImportError):
            try:
                return torch._C._StreamBase
            except (AttributeError, ImportError):
                log_warning(
                    "torch._streambase._StreamBase not found, using object as base class for Stream - "
                    "this may cause issues with PyTorch dynamo"
                )
                return object

_StreamBase = _get_stream_base()

# --- Stream Replacement ---

class Stream(_StreamBase):
    """Replacement for torch.cuda.Stream (MPS/CPU)."""

    @auto_log()
    def __new__(cls, *args, **kwargs):
        # Always allow creation, even if _StreamBase is object
        instance = super().__new__(cls)
        return instance

    @auto_log()
    def __init__(self, *args, **kwargs):
        # Accept device/priority from args or kwargs for PyTorch compatibility
        device = None
        priority = 0
        # Parse device and priority from args
        if len(args) > 0:
            device = args[0]
        if len(args) > 1:
            priority = args[1]
        # Override with kwargs if present
        if 'device' in kwargs:
            device = kwargs['device']
        if 'priority' in kwargs:
            priority = kwargs['priority']
        if _StreamBase is not object:
            try:
                super().__init__()
            except Exception as e:
                log_warning(f"Error calling _StreamBase.__init__: {e}")
        if device is None:
            device = get_default_device()
        self._td_device = torch_device_replacement(device)
        self._td_priority = priority
        self._is_created = True
        self._is_destroyed = False
        self.cuda_stream = self
        self._old_stream = None

    @property
    def device(self):
        return getattr(self, '_td_device', None)

    @property
    def priority(self):
        return getattr(self, '_td_priority', 0)

    @auto_log()
    def synchronize(self):
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        return self

    @auto_log()
    def query(self):
        return True

    @auto_log()
    def wait_event(self, event=None):
        if event is not None:
            event_device = getattr(event, '_device', getattr(event, 'device', None))
            if event_device is not None and event_device != self._td_device:
                log_warning(f"Stream and event device mismatch: {self._td_device} vs {event_device}")
        return self

    @auto_log()
    def wait_stream(self, stream=None):
        if hasattr(stream, 'synchronize'):
            stream.synchronize()
        self.synchronize()
        return self

    @auto_log()
    def record_event(self, event=None):
        if event is None:
            from ..events.cuda_events import Event
            event = Event(enable_timing=True)
        event.record(self)
        return event

    @auto_log()
    def record(self, event=None):
        return self

    @auto_log()
    def wait(self):
        return self

    @property
    def is_completed(self):
        return True

    @property
    def ipc_handle(self):
        return None

    def __enter__(self):
        self._old_stream = torch.cuda.current_stream() if hasattr(torch.cuda, 'current_stream') else None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._old_stream is not None:
            # Restore previous stream if possible (no-op for MPS)
            pass
        return False

    def __del__(self):
        if hasattr(self, '_is_destroyed') and not self._is_destroyed:
            self._is_destroyed = True

    def __str__(self):
        return f"Stream(device={self.device}, priority={self.priority})"

    def __eq__(self, o):
        if isinstance(o, Stream):
            return (self.device == o.device and self.priority == o.priority)
        return False

    def __hash__(self):
        return hash((self.device, self.priority))

# --- Stream Factory Functions ---

def current_stream(device: Optional[Any] = None) -> Stream:
    """Get the current CUDA stream."""
    return Stream(device=device)

def default_stream(device: Optional[Any] = None) -> Stream:
    """Get the default CUDA stream."""
    return Stream(device=device)

def stream(stream: Optional[Stream] = None):
    """Context-manager that selects a given stream."""
    class StreamContext:
        @auto_log()
        def __init__(self, stream):
            self.stream = stream

        @auto_log()
        def __enter__(self):
            if self.stream is not None and hasattr(self.stream, '__enter__'):
                self.stream.__enter__()
            return self.stream

        @auto_log()
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.stream is not None and hasattr(self.stream, '__exit__'):
                return self.stream.__exit__(exc_type, exc_val, exc_tb)
            return False

        @auto_log()
        def query(self):
            if self.stream is not None and hasattr(self.stream, 'query'):
                return self.stream.query()
            return True

        @auto_log()
        def synchronize(self):
            if self.stream is not None and hasattr(self.stream, 'synchronize'):
                return self.stream.synchronize()
            return self

        @auto_log()
        def wait_event(self, event=None):
            if self.stream is not None and hasattr(self.stream, 'wait_event'):
                return self.stream.wait_event(event)
            return self

        @auto_log()
        def wait_stream(self, stream=None):
            if self.stream is not None and hasattr(self.stream, 'wait_stream'):
                return self.stream.wait_stream(stream)
            return self

        @auto_log()
        def record_event(self, event=None):
            if self.stream is not None and hasattr(self.stream, 'record_event'):
                return self.stream.record_event(event)
            return self

    return StreamContext(stream)

def apply_patches():
    """Apply all CUDA stream-related patches."""
    torch.cuda.Stream = Stream
    torch.cuda.current_stream = current_stream
    torch.cuda.default_stream = default_stream
    torch.cuda.stream = stream
    
    # Add global Stream class patch if it exists
    if hasattr(torch, 'Stream'):
        class StreamWrapper:
            def __new__(cls, device=None, *args, **kwargs):
                if device is not None:
                    device = torch_device_replacement(device)
                return Stream(device)
        torch.Stream = StreamWrapper
