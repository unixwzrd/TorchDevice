import torch
from typing import Optional, Any
import psutil
from ..modules.TDLogger import auto_log
import contextlib

# --- Device-related Emulation Functions ---

def _set_device(device: Optional[Any]) -> None:
    """Set the current device (only mps:0 is supported)."""
    if device not in (0, "mps", "mps:0", None):
        pass


def _current_device() -> int:
    """Return the current device index (always 0 for MPS/CPU)."""
    return 0


def _device_count() -> int:
    """Return the number of available devices (1 for MPS, 0 for CPU)."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 1
    return 0


def _is_available() -> bool:
    """Return True if a CUDA/MPS device is available, False otherwise."""
    return (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())


def _get_device_name(device: Optional[Any] = None) -> str:
    """Return the name of the current device."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'Apple MPS'
    return 'CPU'


def _get_device_capability(device: Optional[Any] = None) -> tuple:
    """Return the device capability (mocked for MPS/CPU)."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return (0, 0)
    return (0, 0)


def _get_device_properties(device: Optional[Any] = None):
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


def _is_initialized() -> bool:
    """Return True if CUDA/MPS is initialized."""
    return _is_available()


def _get_arch_list() -> list:
    """Return architecture list (mocked for MPS/CPU)."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return ['mps']
    return []


def _is_built() -> bool:
    """Return True if CUDA/MPS is built (mocked)."""
    return _is_available()


def _cuda_device_context(device=0):
    # On MPS, masquerade as CUDA context manager (no-op)
    yield


# --- Patch Application ---

def apply_patches() -> None:
    import torch
    # Patch CUDA context manager to masquerade as CUDA on MPS
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.cuda.device = contextlib.contextmanager(_cuda_device_context)
        # Patch is_built as a function, not a bool
        if hasattr(torch.backends.cuda, 'is_built'):
            torch.backends.cuda.is_built = lambda: True
        else:
            setattr(torch.backends.cuda, 'is_built', lambda: True)
        torch.cuda.is_available = lambda: True
    torch.cuda.set_device = _set_device
    torch.cuda.current_device = _current_device
    torch.cuda.device_count = _device_count
    torch.cuda.is_available = _is_available
    torch.cuda.get_device_name = _get_device_name
    torch.cuda.get_device_capability = _get_device_capability
    torch.cuda.get_device_properties = _get_device_properties
    torch.cuda.is_initialized = _is_initialized
    torch.cuda.get_arch_list = _get_arch_list
    # Patch torch.cuda.is_built as a function for API consistency
    torch.cuda.is_built = _is_built


# Device-related mock_* functions migrated from TorchDevice.py

# Remove from here:
@auto_log()
def mock_cuda_is_available(cls):
    return cls._default_device in ['cuda', 'mps']


@auto_log()
def mock_cuda_device_count(cls):
    if cls._default_device == 'cuda':
        return cls._original_torch_cuda_device_count()
    elif cls._default_device == 'mps':
        return 1
    else:
        return 0


@auto_log()
def mock_cuda_get_device_properties(cls, device):
    if cls._default_device == 'cuda':
        return cls._original_torch_cuda_get_device_properties(device)
    elif cls._default_device == 'mps':
        class MPSDeviceProperties:
            name = 'Apple MPS'
            total_memory = psutil.virtual_memory().total

            def __str__(self):
                return f'MPSDeviceProperties(name={self.name}, total_memory={self.total_memory})'
        return MPSDeviceProperties()
    else:
        raise RuntimeError("No GPU device available")


@auto_log()
def mock_cuda_is_initialized(cls):
    return cls._default_device in ['cuda', 'mps']


@auto_log()
def mock_cuda_get_arch_list(cls):
    if cls._default_device == 'cuda':
        return cls._original_torch_cuda_get_arch_list()
    elif cls._default_device == 'mps':
        return ['mps']
    else:
        return []


@auto_log()
def mock_cuda_is_built(cls):
    if cls._default_device in ['cuda', 'mps']:
        return True
    else:
        return False


@auto_log()
def mock_cuda_get_device_name(cls, device=None):
    if cls._default_device == 'cuda':
        return cls._original_torch_cuda_get_device_name(device)
    elif cls._default_device == 'mps':
        return 'Apple MPS'
    else:
        return 'CPU'


@auto_log()
def mock_cuda_set_device(cls, device):
    if cls._default_device == 'cuda':
        cls._original_torch_cuda_set_device(device)


@auto_log()
def mock_cuda_synchronize(cls, device=None):
    if cls._default_device == 'cuda':
        cls._original_torch_cuda_synchronize(device)
    elif cls._default_device == 'mps':
        import torch
        torch.mps.synchronize()


@auto_log()
def mock_cuda_get_device_capability(cls, device=None):
    if cls._default_device == 'cuda':
        return cls._original_torch_cuda_get_device_capability(device)
    elif cls._default_device == 'mps':
        return (0, 0)
    else:
        return (0, 0)


@auto_log()
def mock_cuda_ipc_collect(cls):
    if cls._default_device == 'cuda':
        import torch
        return torch.cuda.ipc_collect()


@auto_log()
def mock_cuda_function_stub(cls, *args, **kwargs):
    pass


@auto_log()
def mock_cuda_current_device(cls):
    if cls._default_device == 'cuda':
        return cls._original_torch_cuda_current_device()
    elif cls._default_device == 'mps':
        return 0
    else:
        return -1


@auto_log()
def mock_cuda_device_context(cls, device=None):
    class DeviceContextManager:
        @auto_log()
        def __init__(self, device):
            self.device = device

        @auto_log()
        def __enter__(self):
            cls.mock_cuda_set_device(self.device)

        @auto_log()
        def __exit__(self, exc_type, exc_value, traceback):
            pass
    return DeviceContextManager(device)


@auto_log()
def mock_cuda_empty_cache(cls):
    if cls._default_device == 'cuda':
        cls._original_torch_cuda_empty_cache()
    elif cls._default_device == 'mps':
        import torch
        torch.mps.empty_cache()


# Remove up to here.
# ... existing code ... 