"""
Device Detection
"""
import torch
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
        if torch.cuda.is_available():
            _CACHED_DEFAULT_DEVICE = 'cuda'
            log_info("CUDA device detected and available, using as default device")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _CACHED_DEFAULT_DEVICE = 'mps'
            log_info("MPS device detected and available, using as default device")
        else:
            _CACHED_DEFAULT_DEVICE = 'cpu'
            log_info("No GPU devices available, falling back to CPU")
    return _CACHED_DEFAULT_DEVICE

@auto_log()
def redirect_device_type(device_type, cpu_override=False):
    """
    Redirect a device type string based on availability and CPU override.
    If cpu_override is True, always returns 'cpu'.
    If device_type is 'cuda' or 'mps' and that device is available, returns it.
    Otherwise, falls back to available device.
    """
    if device_type == 'cpu' or cpu_override:
        return 'cpu'
    if device_type in ['cuda', 'mps']:
        if device_type == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        elif device_type == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    # Fallback: prefer CUDA if available, else MPS, else CPU.
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'