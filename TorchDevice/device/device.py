import torch
from typing import Optional, Any
import psutil
from ..modules.TDLogger import auto_log
import contextlib

# --- Device-related Emulation Functions ---

def t_cuda_set_device(device: Optional[Any]) -> None:
    """Set the current device (only mps:0 is supported)."""
    if device not in (0, "mps", "mps:0", None):
        pass


def t_cuda_current_device() -> int:
    """Return the current device index (always 0 for MPS/CPU)."""
    return 0


def t_cuda_device_count() -> int:
    """Return the number of available devices (1 for MPS, 0 for CPU)."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 1
    return 0


def t_cuda_is_available() -> bool:
    """Return True if a CUDA/MPS device is available, False otherwise."""
    return (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())


def t_cuda_get_device_name(device: Optional[Any] = None) -> str:
    """Return the name of the current device."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'Apple MPS'
    return 'CPU'


def t_cuda_get_device_capability(device: Optional[Any] = None) -> tuple:
    """Return the device capability (mocked for MPS/CPU)."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return (0, 0)
    return (0, 0)


def t_cuda_get_device_properties(device: Optional[Any] = None):
    """Return device properties (mocked for MPS/CPU)."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        class MPSDeviceProperties:
            name = 'Apple MPS'
            total_memory = psutil.virtual_memory().total

            def __str__(self):
                return f'MPSDeviceProperties(name={self.name}, total_memory={self.total_memory})'
        return MPSDeviceProperties()
    else:
        raise RuntimeError("No GPU device available")


def t_cuda_is_initialized() -> bool:
    """Return True if CUDA/MPS is initialized."""
    return t_cuda_is_available()


def t_cuda_get_arch_list() -> list:
    """Return architecture list (mocked for MPS/CPU)."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return ['mps']
    return []


def t_cuda_is_built() -> bool:
    """Return True if CUDA/MPS is built (mocked)."""
    return t_cuda_is_available()


def t_cuda_device_context(device=0):
    # On MPS, masquerade as CUDA context manager (no-op)
    yield


# --- Patch Application ---

def apply_patches() -> None:
    import torch
    # Patch CUDA context manager to masquerade as CUDA on MPS
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.cuda.device = contextlib.contextmanager(t_cuda_device_context)
        # Patch is_built as a function, not a bool
        if hasattr(torch.backends.cuda, 'is_built'):
            torch.backends.cuda.is_built = lambda: True
        else:
            setattr(torch.backends.cuda, 'is_built', lambda: True)
        torch.cuda.is_available = lambda: True
    torch.cuda.set_device = t_cuda_set_device
    torch.cuda.current_device = t_cuda_current_device
    torch.cuda.device_count = t_cuda_device_count
    torch.cuda.is_available = t_cuda_is_available
    torch.cuda.get_device_name = t_cuda_get_device_name
    torch.cuda.get_device_capability = t_cuda_get_device_capability
    torch.cuda.get_device_properties = t_cuda_get_device_properties
    torch.cuda.is_initialized = t_cuda_is_initialized
    torch.cuda.get_arch_list = t_cuda_get_arch_list
    # Patch torch.cuda.is_built as a function for API consistency
    torch.cuda.is_built = t_cuda_is_built
