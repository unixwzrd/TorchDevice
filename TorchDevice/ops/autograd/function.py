"""
TorchDevice Autograd Function Module
------------------------------
Autograd function operations and patches.
"""

import torch
from typing import Optional, Any
from TorchDevice.core.logger import log_info, auto_log

# Store original functions
t_grad_fn = getattr(torch.autograd, 'grad_fn', None)
t_Function = getattr(torch.autograd, 'Function', None)


@auto_log()
def grad_fn(fn: Any, *args, **kwargs) -> Optional[Any]:
    """Create a gradient function."""
    if t_grad_fn:
        return t_grad_fn(fn, *args, **kwargs)
    return None


def apply_patches() -> None:
    """Apply autograd function patches."""
    log_info("Applying autograd function patches")
    
    # Patch autograd functions
    if hasattr(torch.autograd, 'grad_fn'):
        setattr(torch.autograd, 'grad_fn', grad_fn)
    
    log_info("Autograd function patches applied")


# Module initialization
log_info("Initializing TorchDevice autograd function module")

__all__: list[str] = [
    'grad_fn',
    'apply_patches'
]

log_info("TorchDevice autograd function module initialized") 