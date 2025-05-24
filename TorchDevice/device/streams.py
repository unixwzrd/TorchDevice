import torch
from typing import Any, Optional  # noqa: F401
import time
from ..modules.TDLogger import auto_log, log_warning
from ..TorchDevice import TorchDevice


# --- Base Class Resolution Helpers ---

def _get_stream_base():
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


def _get_event_base():
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


_StreamBase = _get_stream_base()
_EventBase = _get_event_base()


# --- Stream Replacement ---

class t_cuda_Stream(_StreamBase):
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
            device = TorchDevice.get_default_device()
        self._td_device = torch.device(device)
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
            event = t_cuda_event(enable_timing=True)
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
        return f"t_cuda_Stream(device={self.device}, priority={self.priority})"

    def __eq__(self, o):
        if isinstance(o, t_cuda_Stream):
            return (self.device == o.device and self.priority == o.priority)
        return False

    def __hash__(self):
        return hash((self.device, self.priority))


# --- Event Replacement ---

def t_get_mps_event_class():
    class t_cuda_Event(_EventBase):

        @auto_log()
        def __init__(self, enable_timing=False, blocking=False, interprocess=False, device=None):
            if _EventBase is not object:
                try:
                    super().__init__()
                except Exception as e:
                    log_warning(f"Error calling _EventBase.__init__: {e}")
            if device is None:
                device = TorchDevice.get_default_device()
            self._device = torch.device(device)
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
            start_time = self._record_time
            end_time = getattr(end_event, '_record_time', time.time())
            if start_time is None or end_time is None:
                log_warning("Event timestamps are None, returning mock value")
                return 0.5
            elapsed_ms = (end_time - start_time) * 1000.0
            return elapsed_ms

        @auto_log()
        def synchronize(self):
            if not self._recorded:
                log_warning("Event has not been recorded yet")
            return self

        @auto_log()
        def is_completed(self):
            return self._recorded

        @auto_log()
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
            return f"t_cuda_Event(device={self._device}, timing={self.enable_timing})"

        def __eq__(self, o):
            if isinstance(o, t_cuda_Event):
                return (self._device == o._device and self.enable_timing == o.enable_timing)
            return False

        def __hash__(self):
            return hash((self._device, self.enable_timing))

    return t_cuda_Event


# --- Stream/Event Factory Functions ---

def t_cuda_stream_class(*args, **kwargs):
    # Accepts any combination of device/priority as args or kwargs
    return t_cuda_Stream(*args, **kwargs)


def t_cuda_event(*args, **kwargs):
    enable_timing = kwargs.get('enable_timing', False)
    blocking = kwargs.get('blocking', False)
    interprocess = kwargs.get('interprocess', False)
    device = kwargs.get('device', None)
    t_Event = t_get_mps_event_class()
    return t_Event(enable_timing=enable_timing, blocking=blocking, interprocess=interprocess, device=device)


def t_cuda_stream(stream=None):
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


def t_cuda_current_stream(device=None):
    return t_cuda_stream_class(device=device)


def t_cuda_default_stream(device=None):
    return t_cuda_stream_class(device=device)


def t_cuda_synchronize(device=None):
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize()

# --- Patch Application ---

def apply_patches():
    import torch

    torch.cuda.Stream = t_cuda_stream_class
    torch.cuda.Event = t_get_mps_event_class()
    torch.cuda.current_stream = t_cuda_current_stream
    torch.cuda.default_stream = t_cuda_default_stream
    torch.cuda.synchronize = t_cuda_synchronize
    torch.cuda.stream = t_cuda_stream

    # Add global Stream class patch if it exists
    if hasattr(torch, 'Stream'):
        class StreamWrapper:
            def __new__(cls, device=None, *args, **kwargs):
                if device is not None:
                    device = TorchDevice.torch_device_replacement(device)
                return t_cuda_stream_class(device)

        torch.Stream = StreamWrapper
