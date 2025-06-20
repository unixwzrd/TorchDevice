"""
TorchDevice CPU Operations Module
-----------------------------
CPU-specific operations and patches.
"""

import torch
from TorchDevice.core.logger import log_info, auto_log
import TorchDevice.core.tensors # Import the module for dynamic access
# Tensor movement functions are now in core.tensors

# Store original functions
t_Tensor_numpy = torch.Tensor.numpy

# Tensor and module CPU replacement functions are now in core.tensors module

@auto_log()
def numpy_replacement(tensor):
    """
    Replacement for torch.Tensor.numpy() that moves tensor to CPU first if needed.
    This always needs to go to CPU regardless of device policy since numpy()
    requires CPU tensors.
    """
    # Always move to CPU for numpy conversion - this is a special case
    # that must bypass the device redirection policy
    if tensor.device.type != 'cpu':
        cpu_tensor = TorchDevice.core.tensors.t_Tensor_to(tensor, 'cpu') # Use original .to for numpy()
        return t_Tensor_numpy(cpu_tensor)
    return t_Tensor_numpy(tensor)

def apply_patches() -> None:
    """Apply CPU-specific patches."""
    log_info("Applying CPU patches")
    
    # Store original numpy method if not already stored
    global t_Tensor_numpy
    if not t_Tensor_numpy:
        t_Tensor_numpy = torch.Tensor.numpy
    
    # Only patch numpy method - tensor movement functions are handled in core.tensors
    setattr(torch.Tensor, 'numpy', numpy_replacement)
    
    log_info("CPU patches applied")

# Module initialization
log_info("Initializing TorchDevice CPU operations module")

__all__: list[str] = [
    'numpy_replacement',
    'apply_patches'
]

log_info("TorchDevice CPU operations module initialized") 