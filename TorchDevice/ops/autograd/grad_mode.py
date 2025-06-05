"""
TorchDevice Autograd Grad Mode Module
--------------------------------
Gradient computation mode operations.
"""

import torch
from TorchDevice.core.logger import log_info, auto_log

# Store original functions
t_set_grad_enabled = getattr(torch.autograd, 'set_grad_enabled', None)
t_is_grad_enabled = getattr(torch.autograd, 'is_grad_enabled', None)
t_no_grad = getattr(torch.autograd, 'no_grad', None)
t_enable_grad = getattr(torch.autograd, 'enable_grad', None)


@auto_log()
def set_grad_enabled(mode: bool) -> None:
    """Set gradient computation mode."""
    if t_set_grad_enabled:
        t_set_grad_enabled(mode)


@auto_log()
def is_grad_enabled() -> bool:
    """Check if gradient computation is enabled."""
    if t_is_grad_enabled:
        return t_is_grad_enabled()
    return False


@auto_log()
def no_grad():
    """Context-manager that disables gradient computation."""
    if t_no_grad:
        return t_no_grad()
    return None


@auto_log()
def enable_grad():
    """Context-manager that enables gradient computation."""
    if t_enable_grad:
        return t_enable_grad()
    return None


def apply_patches() -> None:
    """Apply autograd grad mode patches."""
    log_info("Applying autograd grad mode patches")
    
    # Patch grad mode functions
    if hasattr(torch.autograd, 'set_grad_enabled'):
        setattr(torch.autograd, 'set_grad_enabled', set_grad_enabled)
    if hasattr(torch.autograd, 'is_grad_enabled'):
        setattr(torch.autograd, 'is_grad_enabled', is_grad_enabled)
    if hasattr(torch.autograd, 'no_grad'):
        setattr(torch.autograd, 'no_grad', no_grad)
    if hasattr(torch.autograd, 'enable_grad'):
        setattr(torch.autograd, 'enable_grad', enable_grad)
    
    log_info("Autograd grad mode patches applied")


# Module initialization
log_info("Initializing TorchDevice autograd grad_mode module")

__all__: list[str] = [
    'set_grad_enabled',
    'is_grad_enabled',
    'no_grad',
    'enable_grad',
    'apply_patches'
]

log_info("TorchDevice autograd grad_mode module initialized") 