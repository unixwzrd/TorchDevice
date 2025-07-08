"""
TorchDevice MPS Streams Module
-------------------------
MPS stream management and operations.
"""

import torch
from typing import Optional, Any
from ...core.logger import log_info, auto_log

# Store original functions if they exist (MPS might not have stream support yet)
t_mps_current_stream = getattr(torch.mps, 'current_stream', None) if hasattr(torch, 'mps') else None
t_mps_default_stream = getattr(torch.mps, 'default_stream', None) if hasattr(torch, 'mps') else None


@auto_log()
def current_stream(device: Optional[torch.device] = None) -> Optional[Any]:
    """Get the current MPS stream."""
    from ...core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'mps' and t_mps_current_stream:
        return t_mps_current_stream(device)
    return None


@auto_log()
def default_stream(device: Optional[torch.device] = None) -> Optional[Any]:
    """Get the default MPS stream."""
    from ...core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'mps' and t_mps_default_stream:
        return t_mps_default_stream(device)
    return None


def apply_patches() -> None:
    """Apply MPS stream patches."""
    log_info("Applying MPS stream patches")
    
    # Only patch if MPS exists and has stream support
    if hasattr(torch, 'mps'):
        if hasattr(torch.mps, 'current_stream'):
            torch.mps.current_stream = current_stream
        if hasattr(torch.mps, 'default_stream'):
            torch.mps.default_stream = default_stream
    
    log_info("MPS stream patches applied")

# Module initialization
log_info("Initializing TorchDevice MPS streams module")

__all__: list[str] = [
    'current_stream',
    'default_stream',
    'apply_patches'
]

log_info("TorchDevice MPS streams module initialized")