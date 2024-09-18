
import inspect
import logging
import os
import sys
import threading

import psutil  # For memory information
import torch

# Configure logging to output to STDERR
logger = logging.getLogger('TorchDevice')
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('[%(caller_filename)s]: GPU REDIRECT in %(caller_func_name)s line %(caller_lineno)d: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_caller_info():
    """Retrieve caller's filename, line number, and function name."""
    frame = inspect.currentframe()
    outer_frames = inspect.getouterframes(frame)
    # The caller is three frames up in the stack
    caller_frame = outer_frames[3]
    frame_info = inspect.getframeinfo(caller_frame[0])
    return os.path.basename(frame_info.filename), frame_info.lineno, frame_info.function


def log_info(message):
    filename, lineno, func_name = get_caller_info()
    logger.info(message, extra={'caller_filename': filename, 'caller_lineno': lineno, 'caller_func_name': func_name})


def log_warning(message):
    filename, lineno, func_name = get_caller_info()
    logger.warning(message, extra={'caller_filename': filename, 'caller_lineno': lineno, 'caller_func_name': func_name})


def log_error(message):
    filename, lineno, func_name = get_caller_info()
    logger.error(message, extra={'caller_filename': filename, 'caller_lineno': lineno, 'caller_func_name': func_name})

class TorchDevice:

    _default_device = None
    _lock = threading.Lock()

    # Save original torch functions as class attributes
    _original_torch_cuda_is_available = torch.cuda.is_available
    _original_torch_cuda_device_count = torch.cuda.device_count
    _original_torch_cuda_get_device_properties = torch.cuda.get_device_properties
    _original_torch_cuda_empty_cache = torch.cuda.empty_cache
    _original_torch_cuda_synchronize = torch.cuda.synchronize
    _original_torch_cuda_current_device = torch.cuda.current_device
    _original_torch_cuda_set_device = torch.cuda.set_device
    _original_torch_cuda_get_device_name = torch.cuda.get_device_name
    _original_torch_cuda_get_device_capability = torch.cuda.get_device_capability
    _original_torch_cuda_is_initialized = torch.cuda.is_initialized
    _original_torch_cuda_get_arch_list = torch.cuda.get_arch_list
    _original_torch_backends_cuda_is_built = torch.backends.cuda.is_built
    _original_torch_device = torch.device
    _original_torch_cuda_device = torch.cuda.device  # Context manager

    def __init__(self, device_type: str=None , device_index: int=None):

        with self._lock:
            if self._default_device is None:
                self.__class__._detect_default_device()

            if device_type is None:
                device_type = self._default_device

            if isinstance(device_type, str):
                if ':' in device_type:
                    device_type, index = device_type.split(':')
                    device_index = int(index)
                else:
                    device_index = 0 if device_index is None else device_index

                device_type = self.__class__._redirect_device_type(device_type)
                device_str = f"{device_type}:{device_index}"
                self.device = self.__class__._original_torch_device(device_str)
            else:
                # If device_type is already a torch.device or other type
                self.device = self.__class__._original_torch_device(device_type)

    def __repr__(self):
        return repr(self.device)

    def __str__(self):
        return str(self.device)

    @classmethod
    def _detect_default_device(cls):
        if cls._original_torch_cuda_is_available():
            cls._default_device = 'cuda'
        elif torch.backends.mps.is_available():
            cls._default_device = 'mps'
        else:
            cls._default_device = 'cpu'

    @classmethod
    def _redirect_device_type(cls, device_type):
        if device_type.startswith('cuda'):
            if cls._original_torch_cuda_is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        elif device_type.startswith('mps'):
            if torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        else:
            # For 'cpu' or other devices, return as is
            return device_type

    # Delegate attribute access to the internal torch.device object
    def __getattr__(self, attr):
        return getattr(self.device, attr)

    # Replace torch.device with our TorchDevice class
    @classmethod
    def torch_device_replacement(cls, device_type=None, device_index=None):
        return cls(device_type, device_index).device

    # Mock and override torch.cuda functions to simulate CUDA on MPS
    @classmethod
    def mock_cuda_is_available(cls):
        """Replacement for torch.cuda.is_available."""
        if cls._original_torch_cuda_is_available():
            log_info("CUDA is available.")
            return True
        elif torch.backends.mps.is_available():
            log_warning("CUDA is not available. Reporting True because MPS is available.")
            return True
        else:
            log_warning("CUDA is not available.")
            return False

    @classmethod
    def mock_cuda_device_count(cls):
        """Replacement for torch.cuda.device_count."""
        if cls._original_torch_cuda_is_available():
            count = cls._original_torch_cuda_device_count()
            log_info(f"CUDA device count: {count}")
            return count
        elif torch.backends.mps.is_available():
            log_warning("CUDA device count requested but CUDA is not available. Returning 1 due to MPS.")
            return 1
        else:
            log_warning("CUDA device count requested but no GPU is available. Returning 0.")
            return 0

    @classmethod
    def mock_cuda_get_device_properties(cls, device):
        """Replacement for torch.cuda.get_device_properties."""
        if cls._original_torch_cuda_is_available():
            props = cls._original_torch_cuda_get_device_properties(device)
            log_info(f"CUDA device properties for device {device}: {props}")
            return props
        elif torch.backends.mps.is_available():
            log_warning("torch.cuda.get_device_properties called but CUDA is not available. Returning MPS properties.")
            # Mock MPS device properties
            class MPSDeviceProperties:
                name = 'Apple MPS'
                total_memory = psutil.virtual_memory().total  # Total system memory

                # Add other attributes as needed
                def __str__(self):
                    return f'MPSDeviceProperties(name={self.name}, total_memory={self.total_memory})'

            return MPSDeviceProperties()
        else:
            log_error("torch.cuda.get_device_properties called but no GPU is available.")
            raise RuntimeError("No GPU device available")

    @classmethod
    def mock_cuda_memory_allocated(cls, device=None):
        """Replacement for torch.cuda.memory_allocated."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        memory_used = mem_info.rss  # Resident Set Size
        return memory_used

    @classmethod
    def mock_cuda_memory_reserved(cls, device=None):
        """Replacement for torch.cuda.memory_reserved."""
        total_memory = psutil.virtual_memory().total
        return total_memory

    @classmethod
    def mock_cuda_max_memory_allocated(cls, device=None):
        """Replacement for torch.cuda.max_memory_allocated."""
        return cls.mock_cuda_memory_allocated(device)

    @classmethod
    def mock_cuda_max_memory_reserved(cls, device=None):
        """Replacement for torch.cuda.max_memory_reserved."""
        return cls.mock_cuda_memory_reserved(device)

    @classmethod
    def mock_cuda_empty_cache(cls):
        """Replacement for torch.cuda.empty_cache."""
        if cls._original_torch_cuda_is_available():
            log_info("Clearing CUDA cache.")
            cls._original_torch_cuda_empty_cache()
        elif torch.backends.mps.is_available():
            log_info("CUDA cache clear requested. Redirecting to clear MPS cache.")
            torch.mps.empty_cache()
        else:
            log_warning("No GPU cache to clear.")

    @classmethod
    def mock_cuda_synchronize(cls, device=None):
        """Replacement for torch.cuda.synchronize."""
        if cls._original_torch_cuda_is_available():
            log_info("Synchronizing CUDA.")
            cls._original_torch_cuda_synchronize(device)
        elif torch.backends.mps.is_available():
            log_info("CUDA synchronize requested. Redirecting to MPS synchronize.")
            torch.mps.synchronize()
        else:
            log_warning("No GPU to synchronize.")

    @classmethod
    def mock_cuda_current_device(cls):
        """Replacement for torch.cuda.current_device."""
        if cls._original_torch_cuda_is_available():
            current_device = cls._original_torch_cuda_current_device()
            log_info(f"Current CUDA device: {current_device}")
            return current_device
        elif torch.backends.mps.is_available():
            log_info("CUDA current device requested but CUDA is not available. Returning 0 due to MPS.")
            return 0
        else:
            log_warning("CUDA current device requested but no GPU is available. Returning -1.")
            return -1

    @classmethod
    def mock_cuda_set_device(cls, device):
        """Replacement for torch.cuda.set_device."""
        if cls._original_torch_cuda_is_available():
            log_info(f"Setting CUDA device to {device}")
            cls._original_torch_cuda_set_device(device)
        elif torch.backends.mps.is_available():
            pass
            # log_info("CUDA set device requested. Ignoring as MPS does not support multiple devices.")
        else:
            log_warning("No GPU available to set device.")

    @classmethod
    def mock_cuda_get_device_name(cls, device=None):
        """Replacement for torch.cuda.get_device_name."""
        if cls._original_torch_cuda_is_available():
            name = cls._original_torch_cuda_get_device_name(device)
            log_info(f"CUDA device name: {name}")
            return name
        elif torch.backends.mps.is_available():
            log_info("CUDA get_device_name called. Returning 'Apple MPS'")
            return 'Apple MPS'
        else:
            log_warning("No GPU available to get device name.")
            return 'CPU'

    @classmethod
    def mock_cuda_get_device_capability(cls, device=None):
        """Replacement for torch.cuda.get_device_capability."""
        if cls._original_torch_cuda_is_available():
            cap = cls._original_torch_cuda_get_device_capability(device)
            log_info(f"CUDA device capability: {cap}")
            return cap
        elif torch.backends.mps.is_available():
            log_info("CUDA get_device_capability called. Returning (0,0) for MPS")
            return (0, 0)
        else:
            log_warning("No GPU available to get device capability.")
            return (0, 0)

    @classmethod
    def mock_cuda_memory_stats(cls, device=None):
        """Replacement for torch.cuda.memory_stats."""
        return {
            'active.all.current': cls.mock_cuda_memory_allocated(device),
            'reserved_bytes.all.current': cls.mock_cuda_memory_reserved(device),
            # Add other stats as needed
        }

    @classmethod
    def mock_cuda_memory_snapshot(cls):
        """Replacement for torch.cuda.memory_snapshot."""
        log_info("CUDA memory_snapshot called. Returning empty list.")
        return []

    @classmethod
    def mock_cuda_memory_summary(cls, device=None, abbreviated=False):
        """Replacement for torch.cuda.memory_summary."""
        log_info("CUDA memory_summary called.")
        summary = f"Memory Allocated: {cls.mock_cuda_memory_allocated(device)} bytes\n"
        summary += f"Memory Reserved: {cls.mock_cuda_memory_reserved(device)} bytes\n"
        return summary

    @classmethod
    def mock_cuda_is_initialized(cls):
        """Replacement for torch.cuda.is_initialized."""
        if cls._original_torch_cuda_is_available():
            log_info("CUDA is initialized.")
            return cls._original_torch_cuda_is_initialized()
        elif torch.backends.mps.is_available():
            log_info("CUDA is not initialized but MPS is available. Reporting as initialized.")
            return True
        else:
            log_warning("No GPU available. CUDA is not initialized.")
            return False

    @classmethod
    def mock_cuda_get_arch_list(cls):
        """Replacement for torch.cuda.get_arch_list."""
        if cls._original_torch_cuda_is_available():
            arch_list = cls._original_torch_cuda_get_arch_list()
            log_info(f"CUDA arch list: {arch_list}")
            return arch_list
        elif torch.backends.mps.is_available():
            log_info("CUDA get_arch_list called. Returning ['mps']")
            return ['mps']
        else:
            log_warning("No GPU available. Returning empty arch list.")
            return []

    @staticmethod
    def mock_cuda_function_stub(*args, **kwargs):
        """Stub function for unsupported CUDA functions."""
        log_warning("This function is called but is not supported on the current hardware. Ignoring.")

    @classmethod
    def mock_cuda_is_built(cls):
        """Replacement for torch.backends.cuda.is_built."""
        if cls._original_torch_backends_cuda_is_built():
            log_info("CUDA backend is built.")
            return True
        elif torch.backends.mps.is_built():
            log_warning("CUDA backend is not built, but MPS backend is built. Reporting as built.")
            return True
        else:
            log_warning("Neither CUDA nor MPS backend is built.")
            return False

    @classmethod
    def mock_cuda_device_context(cls, device=None):
        """Replacement for torch.cuda.device context manager."""
        class DeviceContextManager:
            def __init__(self, device):
                self.device = device

            def __enter__(self):
                cls.mock_cuda_set_device(self.device)

            def __exit__(self, exc_type, exc_value, traceback):
                pass  # No action needed on exit

        return DeviceContextManager(device)

    @classmethod
    def apply_patches(cls):
        """Apply patches to replace torch.device and torch.cuda methods with the mock functions."""
        # List of unsupported functions to be stubbed
        unsupported_functions = [
            'ipc_collect', 'reset_accumulated_memory_stats',
            'reset_peak_memory_stats', 'reset_max_memory_allocated',
            'reset_max_memory_cached', 'stream', 'Stream', 'Event'
        ]

        for func_name in unsupported_functions:
            setattr(torch.cuda, func_name, cls.mock_cuda_function_stub)

        # Override CUDA functions with mocks
        torch.device = cls.torch_device_replacement
        torch.cuda.is_available = cls.mock_cuda_is_available
        torch.cuda.device_count = cls.mock_cuda_device_count
        torch.cuda.get_device_properties = cls.mock_cuda_get_device_properties
        torch.cuda.memory_allocated = cls.mock_cuda_memory_allocated
        torch.cuda.memory_reserved = cls.mock_cuda_memory_reserved
        torch.cuda.max_memory_allocated = cls.mock_cuda_max_memory_allocated
        torch.cuda.max_memory_reserved = cls.mock_cuda_max_memory_reserved
        torch.cuda.empty_cache = cls.mock_cuda_empty_cache
        torch.cuda.synchronize = cls.mock_cuda_synchronize
        torch.cuda.current_device = cls.mock_cuda_current_device
        torch.cuda.set_device = cls.mock_cuda_set_device
        torch.cuda.get_device_name = cls.mock_cuda_get_device_name
        torch.cuda.get_device_capability = cls.mock_cuda_get_device_capability
        torch.cuda.memory_stats = cls.mock_cuda_memory_stats
        torch.cuda.memory_snapshot = cls.mock_cuda_memory_snapshot
        torch.cuda.memory_summary = cls.mock_cuda_memory_summary
        torch.cuda.is_initialized = cls.mock_cuda_is_initialized
        torch.cuda.get_arch_list = cls.mock_cuda_get_arch_list
        torch.backends.cuda.is_built = cls.mock_cuda_is_built
        torch.cuda.device = cls.mock_cuda_device_context  # Override the context manager

        # Apply stubs to any missing CUDA functions
        cuda_functions = unsupported_functions
        for func_name in cuda_functions:
            if not hasattr(torch.cuda, func_name):
                setattr(torch.cuda, func_name, cls.mock_cuda_function_stub)


# Apply patches when the module is imported
# TorchDevice.apply_patches()
# Apply patches on import
class TorchDeviceImporter:
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch":
            self.apply_patches()
        return None

    def apply_patches(self):
        TorchDevice.apply_patches()

sys.meta_path.insert(0, TorchDeviceImporter())
