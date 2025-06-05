"""
TorchDevice CUDA Events Module
-------------------------
CUDA event management and operations.
"""

import torch
from typing import Optional, Any
from TorchDevice.core.logger import log_info, auto_log

from TorchDevice.ops.events.mps_events import MockCudaEvent # Added import

# Store original functions
t_cuda_Event = torch.cuda.Event if hasattr(torch.cuda, 'Event') else None
t_cuda_current_event = getattr(torch.cuda, 'current_event', None)
t_cuda_synchronize = getattr(torch.cuda, 'synchronize', None) # Added original synchronize


@auto_log()
def Event(enable_timing: bool = False, blocking: bool = False, interprocess: bool = False) -> Optional[Any]:
    """Create a new CUDA event or a mock equivalent."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_Event:
        log_info("Returning native CUDA Event for device type: %s", device.type)
        return t_cuda_Event(enable_timing=enable_timing, blocking=blocking, interprocess=interprocess)
    # For MPS or CPU, or if CUDA event is not available (e.g. t_cuda_Event is None but device.type is 'cuda')
    elif device.type in ['mps', 'cpu'] or (device.type == 'cuda' and not t_cuda_Event):
        log_info("Returning MockCudaEvent for device type: %s (t_cuda_Event available: %s)", 
                 device.type, (t_cuda_Event is not None))
        return MockCudaEvent(enable_timing=enable_timing, blocking=blocking, interprocess=interprocess)
    
    log_info("Event creation returning None for device type: %s (this path should ideally not be hit)", device.type)
    return None # Fallback, though above conditions should cover active device types


@auto_log()
def current_event() -> Optional[Any]:
    """Get the current CUDA event."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_current_event is not None:
        return t_cuda_current_event()
    return None


@auto_log()
def torch_cuda_synchronize_replacement(device_arg: Optional[Any] = None) -> None:
    """
    Replacement for torch.cuda.synchronize().
    Redirects to torch.mps.synchronize() if the default device is MPS,
    calls original torch.cuda.synchronize(device_arg) if the default is CUDA,
    or is a no-op if the default is CPU.
    """
    from TorchDevice.core.device import DeviceManager # Local import
    
    default_device_type = DeviceManager.get_default_device().type
    log_info("torch_cuda_synchronize_replacement called with device_arg: %s. Default device type: %s", device_arg, default_device_type)

    if default_device_type == 'cuda':
        if t_cuda_synchronize:
            log_info("Calling original t_cuda_synchronize for CUDA default device.")
            t_cuda_synchronize(device_arg) # Pass original device_arg
        else:
            log_info("Original t_cuda_synchronize is None (CUDA default). No-op.")
    elif default_device_type == 'mps':
        if hasattr(torch.mps, 'synchronize'):
            log_info("Calling torch.mps.synchronize for MPS default device.")
            torch.mps.synchronize() # torch.mps.synchronize does not take a device argument
        else:
            log_info("torch.mps.synchronize not found (MPS default). No-op.")
    elif default_device_type == 'cpu':
        log_info("torch.cuda.synchronize is a no-op on CPU default device.")
        pass # No-op for CPU
    else:
        log_info("Unknown default device type for synchronize: %s", default_device_type)


def apply_patches() -> None:
    """Apply CUDA event and synchronize patches."""
    log_info("Applying CUDA event and synchronize patches")

    # Patch event functions
    if hasattr(torch.cuda, 'Event'): # If torch.cuda.Event exists, patch it
        setattr(torch.cuda, 'Event', Event)
        log_info("Patched torch.cuda.Event.")
    elif hasattr(torch, 'cuda'): # If torch.cuda exists but Event doesn't, create Event pointing to our factory
        setattr(torch.cuda, 'Event', Event)
        log_info("torch.cuda.Event not found, created and patched.")

    if hasattr(torch.cuda, 'current_event'): # If torch.cuda.current_event exists, patch it
        setattr(torch.cuda, 'current_event', current_event)
        log_info("Patched torch.cuda.current_event.")
    elif hasattr(torch, 'cuda'): # If torch.cuda exists but current_event doesn't, create current_event
        setattr(torch.cuda, 'current_event', current_event)
        log_info("torch.cuda.current_event not found, created and patched.")
    
    # Patch synchronize function
    if hasattr(torch, 'cuda'): # Ensure the cuda module itself exists
        setattr(torch.cuda, 'synchronize', torch_cuda_synchronize_replacement)
        log_info("Set torch.cuda.synchronize to torch_cuda_synchronize_replacement.")
    else:
        log_info("torch.cuda module not found, cannot patch synchronize.")

    log_info("CUDA event and synchronize patches applied")


# Module initialization
log_info("Initializing TorchDevice CUDA events module")

__all__: list[str] = [
    'Event',
    'current_event',
    'apply_patches'
]

log_info("TorchDevice CUDA events module initialized")