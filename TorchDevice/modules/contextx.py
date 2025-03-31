import contextlib
import torch
from .modules.device_detection import get_default_device

@contextlib.contextmanager
def cuda_device_context(device=None):
    """
    A context manager that temporarily sets the CUDA device.
    If no device is provided, uses the default device from get_default_device().
    """
    original = torch.cuda.current_device() if torch.cuda.is_available() else None
    target = device or get_default_device()
    torch.cuda.set_device(target)
    try:
        yield
    finally:
        if original is not None:
            torch.cuda.set_device(original)