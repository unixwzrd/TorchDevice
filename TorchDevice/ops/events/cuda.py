"""
TorchDevice CUDA Events Module
-------------------------
CUDA event management and operations. This module now contains the primary
implementation for CUDA events, ported from the original TorchDevice.
"""

import torch
from typing import Optional, Any, Type # Added Type
import time # Added time

from ...core.logger import log_info, auto_log
from ...core.device import DeviceManager # Added DeviceManager

# --- Base Class Resolution Helper (from original streams.py) ---
def _get_event_base() -> Type:
    try:
        from torch._streambase import _EventBase
        return _EventBase
    except (AttributeError, ImportError):
        try:
            from torch._C import _EventBase
            return _EventBase
        except (AttributeError, ImportError):
            try:
                return torch._C._EventBase # type: ignore
            except (AttributeError, ImportError):
                log_warning(
                    "[TORCHDEVICE] torch._streambase._EventBase not found, using object as base class for Event. "
                    "This may cause issues with PyTorch dynamo."
                )
                return object

_EventBase: Type = _get_event_base()

# --- t_cuda_Event Class Definition (from original streams.py) ---
class t_cuda_Event(_EventBase): # type: ignore
    """Replacement for torch.cuda.Event (MPS/CPU)."""
    def __init__(self, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False, device: Optional[Any] = None):
        if _EventBase is not object:
            try:
                super().__init__(enable_timing=enable_timing, blocking=blocking, interprocess=interprocess)
            except TypeError:
                try: # Older PyTorch might not take these in _EventBase init
                    super().__init__()
                except Exception as e_super:
                    log_warning("[TORCHDEVICE] Error calling _EventBase.__init__ for t_cuda_Event: %s", e_super) 
            except Exception as e:
                log_warning("[TORCHDEVICE] Error calling _EventBase.__init__ with args for t_cuda_Event: %s", e)

        actual_device_str = DeviceManager.get_default_device() if device is None else device
        self._device: torch.device = torch.device(DeviceManager.torch_device_replacement(actual_device_str))
        self.enable_timing: bool = enable_timing
        self.blocking: bool = blocking # Note: 'blocking' behavior is specific to CUDA host-side sync
        self.interprocess: bool = interprocess
        if interprocess:
            log_warning("[TORCHDEVICE] Interprocess events not fully supported for non-CUDA devices.")
        
        self._recorded: bool = False
        self._record_time: Optional[float] = None
        self._stream: Optional['torch.cuda.Stream'] = None # Type hint for stream, using torch.cuda.Stream for broader compatibility

    @property
    def device(self) -> torch.device:
        return self._device

    @auto_log()
    def record(self, stream: Optional['torch.cuda.Stream'] = None) -> None: 
        if stream is not None:
             if self.device != stream.device: # type: ignore
                 log_warning("[TORCHDEVICE] Event (%s) and recording stream (%s) device mismatch on record.", self.device, stream.device) # type: ignore
        self._stream = stream 

        self._recorded = True
        if self.enable_timing:
            self._record_time = time.time()

    @auto_log()
    def wait(self, stream: Optional['torch.cuda.Stream'] = None) -> None: 
        if not self._recorded:
            log_warning("[TORCHDEVICE] t_cuda_Event.wait() called before record(). This is typically an error.")
            return

        if self._stream is not None:
            self._stream.synchronize() 
        elif self.blocking:
            if self.device.type == 'mps' and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()

    @auto_log()
    def query(self) -> bool:
        return self._recorded 

    @auto_log()
    def elapsed_time(self, end_event: 't_cuda_Event') -> float:
        if not self.enable_timing or not end_event.enable_timing:
            raise RuntimeError("Timing not enabled for one or both events.")
        if not self._recorded or not end_event._recorded:
            raise RuntimeError("One or both events not recorded.")
        if self._record_time is None or end_event._record_time is None:
            raise RuntimeError("Internal error: record time not set despite timing enabled and event recorded.")
        if self._device != end_event._device:
            log_warning("[TORCHDEVICE] elapsed_time called on events from different devices: %s vs %s", self._device, end_event._device)
        
        return (end_event._record_time - self._record_time) * 1000.0

    @auto_log()
    def synchronize(self) -> None:
        if not self._recorded:
            log_warning("[TORCHDEVICE] t_cuda_Event.synchronize() called before record().")
            return

        if self._stream is not None:
            self._stream.synchronize()
        else:
            if self.device.type == 'mps' and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()

    def __str__(self) -> str:
        return f"<t_cuda_Event device={self.device} recorded={self._recorded}>"

# --- Factory Functions ---
@auto_log()
def t_cuda_event_factory(**kwargs: Any) -> t_cuda_Event:
    """Factory for creating t_cuda_Event instances."""    
    return t_cuda_Event(**kwargs)

@auto_log()
def Event(enable_timing: bool = False, blocking: bool = False, interprocess: bool = False, device: Optional[Any] = None) -> t_cuda_Event:
    """Create a new CUDA event using the t_cuda_Event implementation."""
    return t_cuda_Event(enable_timing=enable_timing, blocking=blocking, interprocess=interprocess, device=device)

@auto_log()
def current_event() -> Optional[Any]: 
    """Get the current CUDA event. (Currently uses original if available, may need mock)"""
    if hasattr(torch.cuda, 'current_event') and torch.cuda.current_event is not None:
         try:
            return torch.cuda.current_event()
         except Exception as e:
            log_info("Error calling original torch.cuda.current_event: %s", e) 
    return None

def apply_patches() -> None:
    """Apply CUDA event patches using t_cuda_Event."""
    log_info("Applying CUDA event patches (ops.events.cuda)")

    if hasattr(torch, 'cuda'):
        setattr(torch.cuda, 'Event', Event) 
        log_info("Patched torch.cuda.Event with t_cuda_Event via factory.")

        log_info("Skipping torch.cuda.current_event patch by ops.events.cuda for now.")
        
        log_info("Skipping torch.cuda.synchronize patch by ops.events.cuda (handled by ops.streams.sync).")
    else:
        log_info("torch.cuda module not found, cannot patch Event for ops.events.cuda.")

    log_info("CUDA event patches applied (ops.events.cuda)")

# Module initialization
log_info("Initializing TorchDevice CUDA events module (ops.events.cuda)")

__all__: list[str] = [
    'Event',        
    'current_event',
    't_cuda_Event', 
    't_cuda_event_factory',
    'apply_patches'
]

log_info("TorchDevice CUDA events module initialized (ops.events.cuda)")