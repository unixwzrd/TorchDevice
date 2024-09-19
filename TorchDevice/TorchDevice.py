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
# Update the formatter to include program name, module, class, function, and line number
formatter = logging.Formatter(
    '[%(program_name)s]: GPU REDIRECT in %(module_name)s.%(class_name)s.%(caller_func_name)s '
    'line %(caller_lineno)d: %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def get_caller_info():
    """Retrieve caller's program name, module name, class name, line number, and function name."""
    frame = inspect.currentframe()
    outer_frames = inspect.getouterframes(frame)
    program_name = os.path.basename(sys.argv[0])  # The main script name
    # Skip frames related to logging and TorchDevice internal calls
    for outer_frame in outer_frames[1:]:
        frame_info = inspect.getframeinfo(outer_frame[0])
        filename = frame_info.filename
        # Exclude frames from TorchDevice and built-in libraries
        excluded_files = ['torchdevice', 'logging', 'importlib', 'threading']
        if not any(excluded in filename.lower() for excluded in excluded_files):
            caller_filename = os.path.basename(filename)
            lineno = frame_info.lineno
            func_name = frame_info.function
            module = inspect.getmodule(outer_frame[0])
            module_name = module.__name__ if module else 'UnknownModule'
            # Attempt to get the class name
            cls_name = 'N/A'
            if 'self' in outer_frame[0].f_locals:
                cls_name = outer_frame[0].f_locals['self'].__class__.__name__
            elif 'cls' in outer_frame[0].f_locals:
                cls_name = outer_frame[0].f_locals['cls'].__name__
            return program_name, module_name, caller_filename, cls_name, lineno, func_name
    # Fallback if no external caller is found
    return program_name, 'UnknownModule', 'UnknownFile', 'N/A', 0, 'UnknownFunction'

def log_info(message, torch_function=None):
    program_name, module_name, filename, cls_name, lineno, func_name = get_caller_info()
    if torch_function:
        message = f"{torch_function} called. {message}"
    logger.info(
        message,
        extra={
            'program_name': program_name,
            'module_name': module_name,
            'caller_filename': filename,
            'class_name': cls_name,
            'caller_lineno': lineno,
            'caller_func_name': func_name
        }
    )

def log_warning(message, torch_function=None):
    program_name, module_name, filename, cls_name, lineno, func_name = get_caller_info()
    if torch_function:
        message = f"{torch_function} called. {message}"
    logger.warning(
        message,
        extra={
            'program_name': program_name,
            'module_name': module_name,
            'caller_filename': filename,
            'class_name': cls_name,
            'caller_lineno': lineno,
            'caller_func_name': func_name
        }
    )

def log_error(message, torch_function=None):
    program_name, module_name, filename, cls_name, lineno, func_name = get_caller_info()
    if torch_function:
        message = f"{torch_function} called. {message}"
    logger.error(
        message,
        extra={
            'program_name': program_name,
            'module_name': module_name,
            'caller_filename': filename,
            'class_name': cls_name,
            'caller_lineno': lineno,
            'caller_func_name': func_name
        }
    )

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

    def __init__(self, device_type: str = None, device_index: int = None):
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
                log_info(f"Creating torch.device('{device_str}')", torch_function='torch.device')
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
            if cls._default_device == 'cuda':
                return 'cuda'
            elif cls._default_device == 'mps':
                log_warning("CUDA device requested but not available. Redirecting to MPS.", torch_function='torch.device')
                return 'mps'
            else:
                log_warning("CUDA device requested but not available. Redirecting to CPU.", torch_function='torch.device')
                return 'cpu'
        elif device_type.startswith('mps'):
            if cls._default_device == 'mps':
                return 'mps'
            elif cls._default_device == 'cuda':
                log_warning("MPS device requested but not available. Redirecting to CUDA.", torch_function='torch.device')
                return 'cuda'
            else:
                log_warning("MPS device requested but not available. Redirecting to CPU.", torch_function='torch.device')
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
        if cls._default_device in ['cuda', 'mps']:
            log_info("CUDA is available.", torch_function='torch.cuda.is_available')
            return True
        else:
            log_warning("CUDA is not available.", torch_function='torch.cuda.is_available')
            return False

    @classmethod
    def mock_cuda_device_count(cls):
        """Replacement for torch.cuda.device_count."""
        if cls._default_device == 'cuda':
            count = cls._original_torch_cuda_device_count()
            log_info(f"CUDA device count: {count}", torch_function='torch.cuda.device_count')
            return count
        elif cls._default_device == 'mps':
            log_info("Returning device count as 1 for MPS.", torch_function='torch.cuda.device_count')
            return 1
        else:
            log_warning("CUDA device count requested but no GPU is available. Returning 0.", torch_function='torch.cuda.device_count')
            return 0

    @classmethod
    def mock_cuda_get_device_properties(cls, device):
        """Replacement for torch.cuda.get_device_properties."""
        if cls._default_device == 'cuda':
            props = cls._original_torch_cuda_get_device_properties(device)
            log_info(f"CUDA device properties for device {device}: {props}", torch_function='torch.cuda.get_device_properties')
            return props
        elif cls._default_device == 'mps':
            log_info("Returning MPS device properties.", torch_function='torch.cuda.get_device_properties')
            # Mock MPS device properties
            class MPSDeviceProperties:
                name = 'Apple MPS'
                total_memory = psutil.virtual_memory().total  # Total system memory

                # Add other attributes as needed
                def __str__(self):
                    return f'MPSDeviceProperties(name={self.name}, total_memory={self.total_memory})'

            return MPSDeviceProperties()
        else:
            log_error("No GPU device available.", torch_function='torch.cuda.get_device_properties')
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
        if cls._default_device == 'cuda':
            log_info("Clearing CUDA cache.", torch_function='torch.cuda.empty_cache')
            cls._original_torch_cuda_empty_cache()
        elif cls._default_device == 'mps':
            log_info("Clearing MPS cache.", torch_function='torch.cuda.empty_cache')
            torch.mps.empty_cache()
        else:
            log_warning("No GPU cache to clear.", torch_function='torch.cuda.empty_cache')

    @classmethod
    def mock_cuda_synchronize(cls, device=None):
        """Replacement for torch.cuda.synchronize."""
        if cls._default_device == 'cuda':
            log_info("Synchronizing CUDA.", torch_function='torch.cuda.synchronize')
            cls._original_torch_cuda_synchronize(device)
        elif cls._default_device == 'mps':
            log_info("Synchronizing MPS.", torch_function='torch.cuda.synchronize')
            torch.mps.synchronize()
        else:
            log_warning("No GPU to synchronize.", torch_function='torch.cuda.synchronize')

    @classmethod
    def mock_cuda_current_device(cls):
        """Replacement for torch.cuda.current_device."""
        if cls._default_device == 'cuda':
            current_device = cls._original_torch_cuda_current_device()
            log_info(f"Current CUDA device: {current_device}", torch_function='torch.cuda.current_device')
            return current_device
        elif cls._default_device == 'mps':
            log_info("Returning current MPS device (0).", torch_function='torch.cuda.current_device')
            return 0
        else:
            log_warning("No GPU available. Returning -1.", torch_function='torch.cuda.current_device')
            return -1

    @classmethod
    def mock_cuda_set_device(cls, device):
        """Replacement for torch.cuda.set_device."""
        if cls._default_device == 'cuda':
            log_info(f"Setting CUDA device to {device}", torch_function='torch.cuda.set_device')
            cls._original_torch_cuda_set_device(device)
        elif cls._default_device == 'mps':
            log_warning("MPS does not support setting device.", torch_function='torch.cuda.set_device')
            # No action needed; MPS does not support multiple devices
        else:
            log_warning("No GPU available to set device.", torch_function='torch.cuda.set_device')

    @classmethod
    def mock_cuda_get_device_name(cls, device=None):
        """Replacement for torch.cuda.get_device_name."""
        if cls._default_device == 'cuda':
            name = cls._original_torch_cuda_get_device_name(device)
            log_info(f"CUDA device name: {name}", torch_function='torch.cuda.get_device_name')
            return name
        elif cls._default_device == 'mps':
            log_info("Returning 'Apple MPS' as device name.", torch_function='torch.cuda.get_device_name')
            return 'Apple MPS'
        else:
            log_warning("No GPU available to get device name.", torch_function='torch.cuda.get_device_name')
            return 'CPU'

    @classmethod
    def mock_cuda_get_device_capability(cls, device=None):
        """Replacement for torch.cuda.get_device_capability."""
        if cls._default_device == 'cuda':
            cap = cls._original_torch_cuda_get_device_capability(device)
            log_info(f"CUDA device capability: {cap}", torch_function='torch.cuda.get_device_capability')
            return cap
        elif cls._default_device == 'mps':
            log_info("Returning (0, 0) for MPS device capability.", torch_function='torch.cuda.get_device_capability')
            return (0, 0)
        else:
            log_warning("No GPU available to get device capability.", torch_function='torch.cuda.get_device_capability')
            return (0, 0)

    @classmethod
    def mock_cuda_memory_stats(cls, device=None):
        """Replacement for torch.cuda.memory_stats."""
        stats = {
            'active.all.current': cls.mock_cuda_memory_allocated(device),
            'reserved_bytes.all.current': cls.mock_cuda_memory_reserved(device),
            # Add other stats as needed
        }
        log_info(f"Memory stats: {stats}", torch_function='torch.cuda.memory_stats')
        return stats

    @classmethod
    def mock_cuda_memory_snapshot(cls):
        """Replacement for torch.cuda.memory_snapshot."""
        log_info("Returning empty memory snapshot.", torch_function='torch.cuda.memory_snapshot')
        return []

    @classmethod
    def mock_cuda_memory_summary(cls, device=None, abbreviated=False):
        """Replacement for torch.cuda.memory_summary."""
        log_info("Generating memory summary.", torch_function='torch.cuda.memory_summary')
        summary = f"Memory Allocated: {cls.mock_cuda_memory_allocated(device)} bytes\n"
        summary += f"Memory Reserved: {cls.mock_cuda_memory_reserved(device)} bytes\n"
        return summary

    @classmethod
    def mock_cuda_is_initialized(cls):
        """Replacement for torch.cuda.is_initialized."""
        if cls._default_device in ['cuda', 'mps']:
            log_info("CUDA is initialized.", torch_function='torch.cuda.is_initialized')
            return True
        else:
            log_warning("CUDA is not initialized.", torch_function='torch.cuda.is_initialized')
            return False

    @classmethod
    def mock_cuda_get_arch_list(cls):
        """Replacement for torch.cuda.get_arch_list."""
        if cls._default_device == 'cuda':
            arch_list = cls._original_torch_cuda_get_arch_list()
            log_info(f"CUDA arch list: {arch_list}", torch_function='torch.cuda.get_arch_list')
            return arch_list
        elif cls._default_device == 'mps':
            log_info("Returning ['mps'] as arch list.", torch_function='torch.cuda.get_arch_list')
            return ['mps']
        else:
            log_warning("No GPU available. Returning empty arch list.", torch_function='torch.cuda.get_arch_list')
            return []

    @classmethod
    def mock_cuda_is_built(cls):
        """Replacement for torch.backends.cuda.is_built."""
        if cls._default_device == 'cuda':
            log_info("CUDA backend is built.", torch_function='torch.backends.cuda.is_built')
            return True
        elif cls._default_device == 'mps':
            log_warning("CUDA backend is not built, but MPS backend is built. Reporting as built.", torch_function='torch.backends.cuda.is_built')
            return True
        else:
            log_warning("Neither CUDA nor MPS backend is built.", torch_function='torch.backends.cuda.is_built')
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

    # Additional mocks for new functions
    @classmethod
    def mock_cuda_reset_peak_memory_stats(cls):
        """Replacement for torch.cuda.reset_peak_memory_stats."""
        if cls._default_device in ['cuda', 'mps']:
            log_info("Resetting peak memory stats.", torch_function='torch.cuda.reset_peak_memory_stats')
            # Implement any necessary behavior, or provide a no-op
        else:
            log_warning("No GPU available to reset peak memory stats.", torch_function='torch.cuda.reset_peak_memory_stats')

    @classmethod
    def mock_cuda_ipc_collect(cls):
        """Replacement for torch.cuda.ipc_collect."""
        if cls._default_device in ['cuda', 'mps']:
            log_info("Collecting IPC memory.", torch_function='torch.cuda.ipc_collect')
            # Implement any necessary behavior, or provide a no-op
        else:
            log_warning("No GPU available to collect IPC memory.", torch_function='torch.cuda.ipc_collect')

    @classmethod
    def mock_cuda_stream(cls, *args, **kwargs):
        """Replacement for torch.cuda.stream."""
        log_warning("CUDA streams are not supported on this device. Ignoring.", torch_function='torch.cuda.stream')

    @classmethod
    def mock_cuda_stream_class(cls, *args, **kwargs):
        """Replacement for torch.cuda.Stream."""
        log_warning("CUDA Stream class is not supported on this device.", torch_function='torch.cuda.Stream')
        # Return a mock object if necessary
        class MockStream:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                pass

            def __exit__(self, exc_type, exc_value, traceback):
                pass

        return MockStream()

    @classmethod
    def mock_cuda_event(cls, *args, **kwargs):
        """Replacement for torch.cuda.Event."""
        log_warning("CUDA Event is not supported on this device.", torch_function='torch.cuda.Event')
        # Return a mock object if necessary
        class MockEvent:
            def __init__(self, *args, **kwargs):
                pass

            def record(self, *args, **kwargs):
                pass

            def wait(self, *args, **kwargs):
                pass

            def query(self):
                return True

        return MockEvent()

    @staticmethod
    def mock_cuda_function_stub(*args, **kwargs):
        """Stub function for unsupported CUDA functions."""
        log_warning("Unsupported CUDA function called. Ignoring.", torch_function='torch.cuda')

    @classmethod
    def apply_patches(cls):
        """Apply patches to replace torch.device and torch.cuda methods with the mock functions."""
        # Replace torch.device with our TorchDevice class
        torch.device = cls.torch_device_replacement

        # Override CUDA functions with mocks
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

        # Override additional CUDA functions with mocks
        torch.cuda.reset_peak_memory_stats = cls.mock_cuda_reset_peak_memory_stats
        torch.cuda.ipc_collect = cls.mock_cuda_ipc_collect
        torch.cuda.stream = cls.mock_cuda_stream
        torch.cuda.Stream = cls.mock_cuda_stream_class
        torch.cuda.Event = cls.mock_cuda_event

        # List of unsupported functions to be stubbed
        unsupported_functions = [
            'set_stream', 'mem_get_info', 'reset_accumulated_memory_stats',
            'reset_max_memory_allocated', 'reset_max_memory_cached',
            'caching_allocator_alloc', 'caching_allocator_delete',
            'get_allocator_backend', 'change_current_allocator',
            'nvtx', 'jiterator', 'graph', 'CUDAGraph',
            'make_graphed_callables', 'is_current_stream_capturing',
            'graph_pool_handle', 'can_device_access_peer',
            'comm', 'get_gencode_flags', 'current_blas_handle',
            'memory_usage', 'utilization', 'temperature', 'power_draw',
            'clock_rate', 'set_sync_debug_mode', 'get_sync_debug_mode',
            'list_gpu_processes', 'seed', 'seed_all', 'manual_seed',
            'manual_seed_all', 'get_rng_state', 'get_rng_state_all',
            'set_rng_state', 'set_rng_state_all', 'initial_seed',
        ]

        # Apply stubs to unsupported CUDA functions
        for func_name in unsupported_functions:
            if not hasattr(torch.cuda, func_name):
                continue
            setattr(torch.cuda, func_name, cls.mock_cuda_function_stub)

# Apply patches when the module is imported
TorchDevice.apply_patches()