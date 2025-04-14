"""
Device Detection
"""
import torch
from .patching import _original_torch_cuda_is_available
from .TDLogger import log_info, auto_log

# Save the original torch.device type for type checking.
_ORIGINAL_TORCH_DEVICE_TYPE = torch.device("cpu").__class__

# Global cache for default device.
_CACHED_DEFAULT_DEVICE = None

@auto_log()
def get_default_device():
    """
    Return the default device based on available hardware and cache the result.
    Logs which device was chosen.
    """
    global _CACHED_DEFAULT_DEVICE
    if _CACHED_DEFAULT_DEVICE is None:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _CACHED_DEFAULT_DEVICE = 'mps'
            log_info("MPS device detected and available, using as default device")
        elif _original_torch_cuda_is_available():
            _CACHED_DEFAULT_DEVICE = 'cuda'
            log_info("CUDA device detected and available, using as default device")
        else:
            _CACHED_DEFAULT_DEVICE = 'cpu'
            log_info("No GPU devices available, falling back to CPU")

    return _CACHED_DEFAULT_DEVICE


@auto_log()
def redirect_device_type(device_type):
    """
    Redirect a device type string based on availability and CPU override.
    If cpu_override is True, always returns 'cpu'.
    If device_type is 'cuda' or 'mps' and that device is available, returns it.
    Otherwise, falls back to available device.
    """
    if device_type in ['cuda', 'mps']:
        # If MPS is requested and available, use it
        if device_type == 'mps' and torch.backends.mps.is_available():
            device_type = 'mps'
        # If CUDA is requested and available, use it
        elif device_type == 'cuda' and _original_torch_cuda_is_available():
            device_type = 'cuda'
        # If no requested GPU is available, fall back to any available GPU
        elif torch.backends.mps.is_available():
            device_type = 'mps'
        elif _original_torch_cuda_is_available():
            device_type = 'cuda'
        else:
            device_type = 'cpu'

    return device_type
