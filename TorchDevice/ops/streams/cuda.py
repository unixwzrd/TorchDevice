"""
TorchDevice CUDA Streams Module
--------------------------
CUDA stream management and operations.
"""

import torch
from typing import Optional
import torch # Added for torch.device
from typing import Optional, Any, Union # Added for Union type hint
from TorchDevice.core.logger import log_info, log_warning, auto_log # Added log_warning

# Store original functions if they exist
t_cuda_current_stream = getattr(torch.cuda, 'current_stream', None) if hasattr(torch, 'cuda') else None
t_cuda_default_stream = getattr(torch.cuda, 'default_stream', None) if hasattr(torch, 'cuda') else None


class _MPSFallbackCudaStreamMock:
    """A fallback mock for torch.cuda.Stream when operating on an MPS device."""
    def __init__(self, device: torch.device):
        # DO NOT log self!r here as __repr__ might try to access self.device before _actual_device_obj is set.
        log_info(f"BEGIN _MPSFallbackCudaStreamMock.__init__ for device: {device!r}")
        if not isinstance(device, torch.device):
            log_warning(f"_MPSFallbackCudaStreamMock.__init__ received non torch.device: {device!r} (type: {type(device)!r})")
        if device.type != 'mps':
            log_warning(f"_MPSFallbackCudaStreamMock.__init__ initialized with non-MPS device: {device!r}")
        
        self._actual_device_obj = device  # Store the actual device object
        self._id = id(self)  # Basic ID for logging
        log_info(f"_MPSFallbackCudaStreamMock.__init__ (id={self._id}): _actual_device_obj set to {self._actual_device_obj!r}")

    @property
    def device(self) -> torch.device:
        log_info(f"[[ENTERED _MPSFallbackCudaStreamMock.device property GETTER]] self id: {id(self)}, self._id attr: {getattr(self, '_id', 'NOT_FOUND')}") # Explicit entry log

        # Use a temporary variable for self representation in log to avoid recursive __repr__ if it also uses self.device
        obj_repr = f"<TorchDevice._MPSFallbackCudaStreamMock(id={getattr(self, '_id', 'UNKNOWN_ID')})>"
        
        # Check if _actual_device_obj exists to prevent AttributeError during logging itself
        actual_device_val_for_log = getattr(self, '_actual_device_obj', 'MISSING_ATTRIBUTE__actual_device_obj')
        log_info(f"{obj_repr}.device property accessed. Current _actual_device_obj is {actual_device_val_for_log!r}")
        
        if not hasattr(self, '_actual_device_obj'):
            log_warning(f"{obj_repr}.device property: _actual_device_obj IS MISSING! This should not happen.")
            raise AttributeError(f"{obj_repr} does not have _actual_device_obj set. Problem in __init__ or attribute deletion?")
        
        # Ensure what we're returning is indeed a torch.device object
        ret_val = self._actual_device_obj
        if not isinstance(ret_val, torch.device):
            log_warning(f"{obj_repr}.device property: _actual_device_obj is not a torch.device! Type: {type(ret_val)!r}, Value: {ret_val!r}. Raising error.")
            raise TypeError(f"{obj_repr}.device must be a torch.device, but _actual_device_obj is {type(ret_val).__name__}")
            
        log_info(f"{obj_repr}.device property returning: {ret_val!r}")
        return ret_val

    def __getattr__(self, name: str):
        obj_repr = f"<TorchDevice._MPSFallbackCudaStreamMock(id={getattr(self, '_id', 'UNKNOWN_ID')})>"
        log_warning(f"{obj_repr}.__getattr__ called for '{name}'. Attribute not found through normal lookup.")
        
        # This is a fallback. If 'device' is requested here, it means the @property didn't catch it.
        if name == 'device' and hasattr(self, '_actual_device_obj'):
            actual_device_val = self._actual_device_obj
            log_warning(f"{obj_repr}.__getattr__ specifically for 'device', _actual_device_obj exists ({actual_device_val!r}). Returning it.")
            return actual_device_val
            
        raise AttributeError(f"'{type(self).__name__}' object (id={getattr(self, '_id', 'UNKNOWN_ID')}) has no attribute '{name}' (caught by __getattr__)")

    def __repr__(self) -> str:
        # Access self.device (property) carefully. If it raises, this __repr__ will fail.
        try:
            device_repr = repr(self.device)
        except AttributeError:
            device_repr = "<device attribute error>"
        except Exception as e:
            device_repr = f"<device access error: {type(e).__name__}>"
            
        return f"<TorchDevice._MPSFallbackCudaStreamMock(device={device_repr}, id={self._id})>"

    def __enter__(self):
        log_info(f"{self!r}.__enter__() called")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log_info(f"{self!r}.__exit__(exc_type={exc_type}, exc_val={exc_val}) called")
        pass

    def query(self) -> bool:
        log_info(f"{self!r}.query() called")
        return True

    def synchronize(self) -> None:
        log_info(f"{self!r}.synchronize() called")
        pass

    def record_event(self, event=None):
        log_info(f"{self!r}.record_event(event={event}) called")
        if event is None:
            # Access self.device (property) to ensure it's used
            current_device = self.device 
            log_warning(f"{self!r}.record_event() called without event on device {current_device!r}; proper mock event creation not implemented.")
            # In a full implementation, this might return a _MPSFallbackCudaEventMock(device=current_device)
            raise NotImplementedError(f"{self!r}.record_event() without event argument is not fully mocked.")
        return event

    def wait_event(self, event) -> None:
        log_info(f"{self!r}.wait_event(event={event}) called")
        if hasattr(event, 'synchronize'):
            event.synchronize()
        else:
            log_warning(f"{self!r}.wait_event() called with event lacking synchronize: {event}")

    @property
    def stream_id(self) -> int:
        log_info(f"{self!r}.stream_id property accessed, returning id {self._id}")
        return self._id

    @property
    def cuda_stream(self) -> int: 
        # Access self.device (property) to ensure the getter is invoked and logged
        device_obj = self.device 
        log_info(f"{self!r}.cuda_stream property accessed (device is {device_obj!r}), returning id {self._id}")
        return self._id # Return the mock stream's own ID as a stand-in for a pointer


def current_stream(device: Optional[torch.device] = None) -> Optional[Union[torch.cuda.Stream, _MPSFallbackCudaStreamMock, Any]]: # Adjusted return type
    print(f">>>>> TorchDevice.ops.streams.cuda.current_stream VERY EARLY ENTRY: device_arg={device!r}, type={type(device)!r}") # DEBUG
    """Get the current CUDA stream, or an MPS equivalent/mock if on an MPS device."""
    # ***** DEBUG PRINT START *****
    print(f"[DEBUG_PRINT] TorchDevice.ops.streams.cuda.current_stream: ENTER. device_arg={device!r}, type(device_arg)={type(device)}")
    # ***** DEBUG PRINT END *****

    # Local imports to prevent circular dependencies and ensure modules are loaded.
    from TorchDevice.core.device import DeviceManager
    
    effective_device = device or DeviceManager.get_default_device()

    if effective_device.type == 'cuda':
        if t_cuda_current_stream:
            return t_cuda_current_stream(effective_device)
        else:
            log_warning("CUDA device type selected, but original torch.cuda.current_stream not found. Returning None.")
            return None
    elif effective_device.type == 'mps':
        # ***** DEBUG PRINT START *****
        print(f"[DEBUG_PRINT] TorchDevice.ops.streams.cuda.current_stream: MPS block. effective_device={effective_device!r}, type(effective_device)={type(effective_device)}")
        # ***** DEBUG PRINT END *****
        log_info(f"TorchDevice: CUDA current_stream call intercepted for MPS device ({effective_device}).")
        # Consistently return _MPSFallbackCudaStreamMock for MPS devices
        # to ensure compatibility with CUDA stream context managers that expect a .device attribute.
        log_info(f"TorchDevice: Consistently returning _MPSFallbackCudaStreamMock for MPS device {effective_device}.")
        # ***** DEBUG PRINT START *****
        print(f"[DEBUG_PRINT] TorchDevice.ops.streams.cuda.current_stream: MPS block. Now consistently returning _MPSFallbackCudaStreamMock for {effective_device!r}.")
        # ***** DEBUG PRINT END *****
        return _MPSFallbackCudaStreamMock(device=effective_device)
        
    elif effective_device.type == 'cpu':
        log_info("TorchDevice: CUDA current_stream call on CPU device. Returning None (no current stream on CPU for CUDA ops).")
        return None
    else:
        log_warning(f"TorchDevice: CUDA current_stream call on unknown device type {effective_device.type}. Returning None.")
        return None


@auto_log()
def default_stream(device: Optional[torch.device] = None) -> Optional[torch.cuda.Stream]:
    """Get the default CUDA stream."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_default_stream:
        return t_cuda_default_stream(device)
    return None


def apply_patches() -> None:
    """Apply CUDA stream patches."""
    log_info("Applying CUDA stream patches")
    
    # Patch stream functions
    if hasattr(torch.cuda, 'current_stream'):
        torch.cuda.current_stream = current_stream
    if hasattr(torch.cuda, 'default_stream'):
        torch.cuda.default_stream = default_stream
    
    log_info("CUDA stream patches applied")

# Module initialization
log_info("Initializing TorchDevice CUDA streams module")

__all__: list[str] = [
    'current_stream',
    'default_stream',
    'apply_patches'
]

log_info("TorchDevice CUDA streams module initialized")