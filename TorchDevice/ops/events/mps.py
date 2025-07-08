"""
TorchDevice MPS Events Module
------------------------
MPS event management and operations.
"""

import torch
from typing import Optional, Any
from ...core.logger import log_info, auto_log

# Store original functions if they exist (MPS might not have event support yet)
t_mps_Event = getattr(torch.mps, 'Event', None) if hasattr(torch, 'mps') else None
t_mps_current_event = getattr(torch.mps, 'current_event', None) if hasattr(torch, 'mps') else None


@auto_log()
def Event(enable_timing: bool = False, blocking: bool = False, interprocess: bool = False) -> Optional[Any]:
    """Create a new MPS event."""
    from ...core.device import DeviceManager  # Local import
    device = DeviceManager.get_default_device()
    if device.type == 'mps' and t_mps_Event:
        return t_mps_Event(enable_timing=enable_timing)
    return None


@auto_log()
def current_event() -> Optional[Any]:
    """Get the current MPS event."""
    from ...core.device import DeviceManager  # Local import
    device = DeviceManager.get_default_device()
    if device.type == 'mps' and t_mps_current_event is not None:
        return t_mps_current_event()
    return None


def apply_patches() -> None:
    """Apply MPS event patches."""
    log_info("Applying MPS event patches")

    # Only patch if MPS exists and has event support
    if hasattr(torch, 'mps'):
        if hasattr(torch.mps, 'Event'):
            setattr(torch.mps, 'Event', Event)
        if hasattr(torch.mps, 'current_event'):
            setattr(torch.mps, 'current_event', current_event)

    log_info("MPS event patches applied")


# Module initialization
log_info("Initializing TorchDevice MPS events module")

__all__: list[str] = [
    'Event',
    'current_event',
    'apply_patches'
]

log_info("TorchDevice MPS events module initialized")