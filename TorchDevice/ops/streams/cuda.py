"""
TorchDevice CUDA Streams and Events Module (ops.streams.cuda)
-----------------------------------------------------------
This module provides mock implementations and wrappers for CUDA stream and
event operations, enabling CUDA-like code execution on MPS or CPU devices,
or intercepting CUDA calls when CUDA is available.

This implementation is a direct port from the original TorchDevice monolithic
codebase (TorchDevice.original.device.streams), adapted for the new
modular structure.
"""
import torch
from typing import Any, Optional, Type, Callable, Dict # noqa: F401
import time

from ...core.logger import auto_log, log_warning, log_info
from ...core.device import DeviceManager


_patches_applied: bool = False

# --- Base Class Resolution Helpers ---
def _get_stream_base() -> Type:
    try:
        from torch._streambase import _StreamBase
        return _StreamBase
    except (AttributeError, ImportError):
        try:
            from torch._C import _StreamBase
            return _StreamBase
        except (AttributeError, ImportError):
            try:
                return torch._C._StreamBase # type: ignore
            except (AttributeError, ImportError):
                log_warning(
                    "[TORCHDEVICE] torch._streambase._StreamBase not found, using object as base class for Stream. "
                    "This may cause issues with PyTorch dynamo."
                )
                return object


_StreamBase: Type = _get_stream_base()

# --- Stream and Event Class Definitions (Ported and Adapted) ---
class t_cuda_Stream(_StreamBase): # type: ignore
    """Replacement for torch.cuda.Stream (MPS/CPU)."""
    def __init__(self, device: Optional[Any] = None, priority: int = 0, **kwargs: Any):
        if _StreamBase is not object:
            try:
                # Pass kwargs for future compatibility e.g. torch.cuda.Stream(..., stream_id=X, device_index=Y)
                super().__init__(device=device, priority=priority, **kwargs)
            except TypeError:
                 try: # Older PyTorch might not take device/priority in _StreamBase init
                     super().__init__(**kwargs)
                 except Exception as e_super:
                     log_warning(f"[TORCHDEVICE] Error calling _StreamBase.__init__ for t_cuda_Stream: {e_super}")       
            except Exception as e:
                log_warning(f"[TORCHDEVICE] Error calling _StreamBase.__init__ with args for t_cuda_Stream: {e}")
        
        actual_device_str = DeviceManager.get_default_device() if device is None else device
        self._td_device: torch.device = torch.device(DeviceManager.torch_device_replacement(actual_device_str))
        self._td_priority: int = priority
        self.cuda_stream: 't_cuda_Stream' = self # For compatibility with some PyTorch internal checks

    @property
    def device(self) -> torch.device:
        return self._td_device

    @property
    def priority(self) -> int:
        return self._td_priority

    @auto_log()
    def synchronize(self) -> None:
        if self._td_device.type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        # For CPU or other types, this is a no-op

    @auto_log()
    def query(self) -> bool:
        return True # For non-CUDA, assume immediate completion

    @auto_log()
    def wait_event(self, event: 'torch.cuda.Event') -> None:
        if event is not None:
            if self.device != event.device:
                log_warning(f"[TORCHDEVICE] Stream ({self.device}) and event ({event.device}) device mismatch on wait_event.")
            event.synchronize() # Simplistic: wait for the event to complete

    @auto_log()
    def wait_stream(self, stream: 't_cuda_Stream') -> None:
        if stream is not None:
            if self.device != stream.device:
                log_warning(f"[TORCHDEVICE] Stream ({self.device}) and waiting stream ({stream.device}) device mismatch on wait_stream.")
            stream.synchronize()

    @auto_log()
    def record_event(self, event: Optional['torch.cuda.Event'] = None) -> 'torch.cuda.Event':
        if event is None:
            event = torch.cuda.Event(device=self.device) # type: ignore[attr-defined]
        if self.device != event.device:
            log_warning(f"[TORCHDEVICE] Stream ({self.device}) and event ({event.device}) device mismatch on record_event.")
            # PyTorch CUDA would error here. We allow it but warn.
        event.record(self)
        return event

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, t_cuda_Stream):
            return False
        return self.device == o.device and self.priority == o.priority # Simplistic, real streams have IDs

    def __hash__(self) -> int:
        return hash((self.device, self.priority))

    def __str__(self) -> str:
        return f"<t_cuda_Stream device={self.device} priority={self.priority}>"

_global_current_streams: Dict[Optional[str], Optional[t_cuda_Stream]] = {}

def _get_current_stream_for_device(device_obj: Optional[torch.device]) -> t_cuda_Stream:
    device_key = device_obj.type if device_obj else DeviceManager.get_default_device().type
    if device_obj and device_obj.index is not None:
        device_key = f"{device_key}:{device_obj.index}"
    
    if device_key not in _global_current_streams or _global_current_streams[device_key] is None:
        default_device_for_stream = device_obj if device_obj else torch.device(DeviceManager.get_default_device())
        _global_current_streams[device_key] = t_cuda_Stream(device=default_device_for_stream)
        #log_info(f"[TORCHDEVICE] Created new default stream for device key {device_key}: {_global_current_streams[device_key]}")
    #else:
        #log_info(f"[TORCHDEVICE] Reusing stream for device key {device_key}: {_global_current_streams[device_key]}")

    return _global_current_streams[device_key] # type: ignore

def _set_current_stream_for_device(stream: Optional[t_cuda_Stream], device_obj: Optional[torch.device]) -> None:
    device_key = device_obj.type if device_obj else DeviceManager.get_default_device().type
    if device_obj and device_obj.index is not None:
        device_key = f"{device_key}:{device_obj.index}"
    _global_current_streams[device_key] = stream
    #log_info(f"[TORCHDEVICE] Set current stream for device key {device_key} to: {stream}")

@auto_log()
def t_cuda_stream_factory(**kwargs: Any) -> t_cuda_Stream:
    """Factory for creating t_cuda_Stream instances."""
    return t_cuda_Stream(**kwargs)

@auto_log()
def t_cuda_stream_context_manager(stream: Optional[t_cuda_Stream]) -> "_StreamContextManager":
    class _StreamContextManager:
        def __init__(self, stream_to_set: Optional[t_cuda_Stream]):
            self.stream_to_set = stream_to_set
            self.prev_stream: Optional[t_cuda_Stream] = None
            self.device_of_context: Optional[torch.device] = None

        def __enter__(self) -> Optional[t_cuda_Stream]:
            if self.stream_to_set is not None:
                self.device_of_context = self.stream_to_set.device
                self.prev_stream = _get_current_stream_for_device(self.device_of_context)
                _set_current_stream_for_device(self.stream_to_set, self.device_of_context)
            return self.stream_to_set

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            if self.stream_to_set is not None and self.device_of_context is not None:
                _set_current_stream_for_device(self.prev_stream, self.device_of_context)

    return _StreamContextManager(stream)

@auto_log()
def t_cuda_current_stream_factory(device: Optional[Any] = None) -> t_cuda_Stream:
    device_obj = torch.device(DeviceManager.torch_device_replacement(device)) if device is not None else None
    return _get_current_stream_for_device(device_obj)

@auto_log()
def t_cuda_default_stream_factory(device: Optional[Any] = None) -> t_cuda_Stream:
    # PyTorch default_stream returns a specific default stream, not necessarily the current one.
    # For our mock, we can return a new stream instance or a cached default per device.
    # Let's return the globally managed current/default stream for that device for simplicity here.
    return t_cuda_current_stream_factory(device)

# --- Patch Application ---
@auto_log()
def apply_patches() -> None:
    global _patches_applied
    if _patches_applied:
        log_info("[TORCHDEVICE] ops.streams.cuda patches already applied.")
        return

    if not hasattr(torch, 'cuda'):
        class MockCudaModule:
            pass
        torch.cuda = MockCudaModule() # type: ignore
        log_info("[TORCHDEVICE] Created mock torch.cuda namespace for stream patching.")
    elif not isinstance(torch.cuda, type(torch)) and not hasattr(torch.cuda, '__path__'): # Check if not a module
        log_warning(f"[TORCHDEVICE] torch.cuda ({type(torch.cuda)}) is not a module. Skipping stream patching for torch.cuda attributes.")
        # We might still want to patch torch.Stream if it exists globally
    else:
        torch.cuda.Stream = t_cuda_Stream # Use the class directly
        torch.cuda.current_stream = t_cuda_current_stream_factory
        torch.cuda.default_stream = t_cuda_default_stream_factory

        # Factory for torch.cuda.stream() context manager
        def _cuda_stream_context_factory(stream_instance=None):
            # If no stream is provided, create a new one using t_cuda_Stream.
            # t_cuda_Stream is expected to handle default device selection if necessary.
            s = stream_instance if stream_instance is not None else t_cuda_Stream()
            return t_cuda_stream_context_manager(s)

        torch.cuda.stream = _cuda_stream_context_factory
        log_info("[TORCHDEVICE] Patched torch.cuda stream attributes.")

    # Add global Stream class patch if it exists (e.g. for older PyTorch versions)
    if hasattr(torch, 'Stream') and not isinstance(torch.Stream, t_cuda_Stream):
        # Be careful not to re-patch if it's already our type from a previous incomplete patch
        # This global torch.Stream is less common in modern PyTorch, torch.cuda.Stream is standard
        class GlobalStreamWrapper:
            def __new__(cls, device: Optional[Any] = None, *args: Any, **kwargs: Any) -> t_cuda_Stream:
                actual_device = device
                if actual_device is not None:
                    actual_device = DeviceManager.torch_device_replacement(actual_device)
                return t_cuda_Stream(device=actual_device, *args, **kwargs)
        torch.Stream = GlobalStreamWrapper # type: ignore
        log_info("[TORCHDEVICE] Patched global torch.Stream.")

    _patches_applied = True
    log_info("[TORCHDEVICE] ops.streams.cuda patches application process complete.")

__all__ = [
    't_cuda_Stream',
    't_cuda_stream_factory',
    't_cuda_stream_context_manager',
    't_cuda_current_stream_factory',
    't_cuda_default_stream_factory',
    'apply_patches'
]
