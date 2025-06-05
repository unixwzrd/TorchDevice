"""
TorchDevice Autograd Variable Module
------------------------------
Autograd variable operations and patches.
"""

import torch
from typing import Optional, Any
from TorchDevice.core.logger import log_info, auto_log

# Store original functions
t_Variable = getattr(torch.autograd, 'Variable', None)
t_is_variable = getattr(torch.autograd, 'is_variable', None)


@auto_log()
def Variable(data: Any, requires_grad: bool = False) -> Optional[Any]:
    """Create a variable that can have gradients."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = DeviceManager.get_default_device()
    if t_Variable:
        var = t_Variable(data, requires_grad=requires_grad)
        return var.to(device) if var is not None else None
    return None


@auto_log()
def is_variable(obj: Any) -> bool:
    """Check if an object is a variable."""
    if t_is_variable:
        return t_is_variable(obj)
    return False


def apply_patches() -> None:
    """Apply autograd variable patches."""
    log_info("Applying autograd variable patches")
    
    # Patch variable functions
    if hasattr(torch.autograd, 'Variable'):
        setattr(torch.autograd, 'Variable', Variable)
    if hasattr(torch.autograd, 'is_variable'):
        setattr(torch.autograd, 'is_variable', is_variable)
    
    log_info("Autograd variable patches applied")


# Module initialization
log_info("Initializing TorchDevice autograd variable module")

__all__: list[str] = [
    'Variable',
    'is_variable',
    'apply_patches'
]

log_info("TorchDevice autograd variable module initialized") 