"""
TorchDevice Event Synchronization Module
----------------------------------
Event synchronization and coordination.
"""

import torch
from typing import Optional, Any
from TorchDevice.core.logger import log_info, auto_log

# Store original functions
t_cuda_wait_event = getattr(torch.cuda, 'wait_event', None)
t_mps_wait_event = getattr(torch.mps, 'wait_event', None) if hasattr(torch, 'mps') else None


@auto_log()
def wait_event(event: Any, device: Optional[torch.device] = None) -> None:
    """Wait for an event to complete."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_wait_event:
        t_cuda_wait_event(event)
    elif device.type == 'mps' and t_mps_wait_event:
        t_mps_wait_event(event)


def apply_patches() -> None:
    """Apply event synchronization patches."""
    log_info("Applying event synchronization patches")
    
    # Patch wait_event functions
    if hasattr(torch.cuda, 'wait_event'):
        torch.cuda.wait_event = wait_event
    if hasattr(torch.mps, 'wait_event'):
        torch.mps.wait_event = wait_event
    
    log_info("Event synchronization patches applied")


# Module initialization
log_info("Initializing TorchDevice event synchronization module")

__all__: list[str] = [
    'wait_event',
    'apply_patches'
]

log_info("TorchDevice event synchronization module initialized") 