import torch
from typing import Any, Optional
import time
from ..modules.TDLogger import auto_log, log_info
from ..TorchDevice import TorchDevice

# --- Stream and Event Replacement Classes/Functions ---

class _cuda_Stream:
    """Replacement for torch.cuda.Stream (MPS/CPU)."""
    @auto_log()
    def __init__(self, device: Optional[Any] = None, priority: int = 0):
        if device is None:
            device = TorchDevice.get_default_device()
        self.device = torch.device(device)
        self.priority = priority
        self._is_created = True
        self._is_destroyed = False

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
            log_info(f"[DEBUG] Stream.wait_event: self.device={self.device}, event.device={event_device}")
            if event_device is not None and event_device != self.device:
                raise RuntimeError(f"Stream and event device mismatch: {self.device} vs {event_device}")
        return self

    @auto_log()
    def wait_stream(self, stream=None):
        return self

    @auto_log()
    def record_event(self, event=None):
        return self

# --- Event Replacement ---

def _get_mps_event_class():
    try:
        from torch._streambase import _EventBase
    except (AttributeError, ImportError):
        try:
            from torch._C import _EventBase
        except (AttributeError, ImportError):
            try:
                _EventBase = torch._C._EventBase
            except (AttributeError, ImportError):
                _EventBase = object

    class _cuda_Event(_EventBase):
        @auto_log()
        def __init__(self, enable_timing=False, blocking=False, interprocess=False, device=None):
            try:
                super().__init__()
            except Exception:
                pass
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
                log_info(f"[DEBUG] Event.record: self._device={self._device}, stream.device={stream_device}")
                if stream_device is not None and self._device != stream_device:
                    raise RuntimeError(f"Event and stream device mismatch: {self._device} vs {stream_device}")
                self._stream = stream
            self._recorded = True
            self._record_time = time.time()
            return self

        @auto_log()
        def wait(self, stream=None):
            return self

        @auto_log()
        def query(self):
            return self._recorded

        @auto_log()
        def elapsed_time(self, end_event):
            if not self.enable_timing:
                return 0.5
            if not self._recorded or not getattr(end_event, '_recorded', False):
                return 0.5
            start_time = self._record_time
            end_time = getattr(end_event, '_record_time', time.time())
            if start_time is None or end_time is None:
                return 0.5
            elapsed_ms = (end_time - start_time) * 1000.0
            return elapsed_ms

        @auto_log()
        def synchronize(self):
            return self

        @auto_log()
        def __del__(self):
            if hasattr(self, '_is_destroyed') and not self._is_destroyed:
                self._is_destroyed = True

    return _cuda_Event

# --- Stream/Event Factory Functions ---

def _cuda_stream_class(device=None, priority=0):
    return _cuda_Stream(device)

def _cuda_event(*args, **kwargs):
    enable_timing = kwargs.get('enable_timing', False)
    blocking = kwargs.get('blocking', False)
    interprocess = kwargs.get('interprocess', False)
    device = kwargs.get('device', None)
    _Event = _get_mps_event_class()
    return _Event(enable_timing=enable_timing, blocking=blocking, interprocess=interprocess, device=device)

def _cuda_stream(stream=None):
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

def _cuda_current_stream(device=None):
    return _cuda_stream_class(device=device)

def _cuda_default_stream(device=None):
    return _cuda_stream_class(device=device)

def _cuda_synchronize(device=None):
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize()

# --- Patch Application ---

def apply_patches():
    import torch
    torch.cuda.Stream = _cuda_stream_class  # type: ignore[assignment]
    torch.cuda.Event = _get_mps_event_class()  # type: ignore[assignment]
    torch.cuda.current_stream = _cuda_current_stream  # type: ignore[assignment]
    torch.cuda.default_stream = _cuda_default_stream  # type: ignore[assignment]
    torch.cuda.synchronize = _cuda_synchronize  # type: ignore[assignment]
    torch.cuda.stream = _cuda_stream  # type: ignore[assignment]
 