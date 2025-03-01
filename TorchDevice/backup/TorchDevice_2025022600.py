import inspect
import logging
import os
import sys
import threading
import psutil  # For memory information
import torch

# --- Capture Original torch.device Type ---
# This must be done before any patches to torch.device occur.
_ORIGINAL_TORCH_DEVICE_TYPE = torch.device("cpu").__class__

# --- Logging Setup and Helpers ---

logger = logging.getLogger('TorchDevice')
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(
    '[%(program_name)s]: GPU REDIRECT in %(module_name)s.%(class_name)s.%(caller_func_name)s '
    'line %(caller_lineno)d: %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

_CURRENT_MODULE = os.path.abspath(__file__)

def get_caller_info():
    """
    Walk the stack and return info from the first frame that is not:
      - from this module,
      - from internal frameworks (e.g. unittest, inspect), or
      - a function known to be part of test harness setup (like setUp/tearDown).
    """
    program_name = os.path.basename(sys.argv[0])
    excluded_functions = {"setUp", "tearDown", "run", "main"}  # add more if needed
    for frame_record in inspect.stack()[1:]:
        abs_filename = os.path.abspath(frame_record.filename)
        # Skip frames from this module
        if abs_filename == _CURRENT_MODULE:
            continue
        # Get module name (if any)
        module = inspect.getmodule(frame_record.frame)
        mod_name = module.__name__ if module else ""
        # Skip internal modules
        if mod_name.startswith("unittest") or mod_name.startswith("inspect"):
            continue
        # Skip known test harness functions
        if frame_record.function in excluded_functions:
            continue
        return {
            'program_name': program_name,
            'module_name': mod_name if mod_name else 'UnknownModule',
            'caller_filename': os.path.basename(abs_filename),
            'class_name': (frame_record.frame.f_locals.get('self').__class__.__name__
                           if 'self' in frame_record.frame.f_locals else 'N/A'),
            'caller_lineno': frame_record.lineno,
            'caller_func_name': frame_record.function
        }
    # Fallback if no candidate is found
    return {
        'program_name': program_name,
        'module_name': 'UnknownModule',
        'caller_filename': 'UnknownFile',
        'class_name': 'N/A',
        'caller_lineno': 0,
        'caller_func_name': 'UnknownFunction'
    }

def log_message(level, message, torch_function=None):
    info = get_caller_info()
    if torch_function:
        message = f"{torch_function} called. {message}"
    logger.log(level, message, extra=info)

def log_info(message, torch_function=None):
    log_message(logging.INFO, message, torch_function)

def log_warning(message, torch_function=None):
    log_message(logging.WARNING, message, torch_function)

def log_error(message, torch_function=None):
    log_message(logging.ERROR, message, torch_function)

# --- Cached Device Detection ---

_CACHED_DEFAULT_DEVICE = None

def get_default_device():
    """Return the default device based on available hardware, using a cached value."""
    global _CACHED_DEFAULT_DEVICE
    if _CACHED_DEFAULT_DEVICE is None:
        if torch.backends.mps.is_available():
            _CACHED_DEFAULT_DEVICE = 'mps'
        elif torch.cuda.is_available():
            _CACHED_DEFAULT_DEVICE = 'cuda'
        else:
            _CACHED_DEFAULT_DEVICE = 'cpu'
    return _CACHED_DEFAULT_DEVICE

# --- PyTorch Function Replacements ---

_in_torch_load = False
_original_torch_load = torch.load
_original_tensor_cuda = torch.Tensor.cuda
_original_module_cuda = torch.nn.Module.cuda
_original_tensor_to = torch.Tensor.to
_original_module_to = torch.nn.Module.to

def tensor_cuda_replacement(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
    default_device = get_default_device()
    log_info(f"tensor.cuda() called with device={device}", "tensor.cuda")
    if default_device == 'mps':
        log_info("Redirecting tensor.cuda() to tensor.to('mps')", "tensor.cuda")
        return self.to('mps', non_blocking=non_blocking, memory_format=memory_format)
    return _original_tensor_cuda(self, device, non_blocking, memory_format)

def module_cuda_replacement(self, device=None):
    default_device = get_default_device()
    log_info(f"nn.Module.cuda() called with device={device}", "nn.Module.cuda")
    if default_device == 'mps':
        log_info("Redirecting nn.Module.cuda() to nn.Module.to('mps')", "nn.Module.cuda")
        return self.to('mps')
    return _original_module_cuda(self, device)

# Intercept .to() calls for both tensors and modules
def tensor_to_replacement(self, *args, **kwargs):
    default_device = get_default_device()
    device_arg = None
    if args:
        candidate = args[0]
        # Check if candidate is a string or a torch.device instance (using the original type),
        # and not a dtype. We use _ORIGINAL_TORCH_DEVICE_TYPE for the isinstance check.
        if isinstance(candidate, (str, _ORIGINAL_TORCH_DEVICE_TYPE)) and (
                not hasattr(candidate, '__module__') or 
                (hasattr(candidate, '__module__') and candidate.__module__ != 'torch')
            ):
            device_arg = candidate
    elif 'device' in kwargs:
        device_arg = kwargs['device']
    if device_arg is not None:
        if isinstance(device_arg, _ORIGINAL_TORCH_DEVICE_TYPE):
            target_device = device_arg.type
        elif isinstance(device_arg, str):
            target_device = device_arg.split(':')[0]
        else:
            target_device = str(device_arg)
        if target_device != default_device:
            log_warning(f"tensor.to() called with device {device_arg} which does not match the default device {default_device}.", "tensor.to")
    return _original_tensor_to(self, *args, **kwargs)

def module_to_replacement(self, *args, **kwargs):
    default_device = get_default_device()
    device_arg = None
    if args:
        candidate = args[0]
        if isinstance(candidate, (str, _ORIGINAL_TORCH_DEVICE_TYPE)) and (
                not hasattr(candidate, '__module__') or 
                (hasattr(candidate, '__module__') and candidate.__module__ != 'torch')
            ):
            device_arg = candidate
    elif 'device' in kwargs:
        device_arg = kwargs['device']
    if device_arg is not None:
        if isinstance(device_arg, _ORIGINAL_TORCH_DEVICE_TYPE):
            target_device = device_arg.type
        elif isinstance(device_arg, str):
            target_device = device_arg.split(':')[0]
        else:
            target_device = str(device_arg)
        if target_device != default_device:
            log_warning(f"module.to() called with device {device_arg} which does not match the default device {default_device}.", "module.to")
    return _original_module_to(self, *args, **kwargs)

torch.Tensor.cuda = tensor_cuda_replacement
torch.nn.Module.cuda = module_cuda_replacement
torch.Tensor.to = tensor_to_replacement
torch.nn.Module.to = module_to_replacement

def torch_load_replacement(*args, **kwargs):
    global _in_torch_load
    if _in_torch_load:
        return _original_torch_load(*args, **kwargs)
    _in_torch_load = True
    try:
        default_device = get_default_device()
        log_info(f"torch.load called with args: {args}, kwargs: {kwargs}", "torch.load")
        if 'map_location' in kwargs:
            if kwargs['map_location'] == 'cpu' or (isinstance(kwargs['map_location'], str) and kwargs['map_location'] != default_device):
                log_info(f"Replacing map_location={kwargs['map_location']} with {default_device}", "torch.load")
                kwargs['map_location'] = default_device
        else:
            log_info(f"Adding map_location={default_device}", "torch.load")
            kwargs['map_location'] = default_device
        return _original_torch_load(*args, **kwargs)
    finally:
        _in_torch_load = False

torch.load = torch_load_replacement

# --- AMP Hooks ---

if hasattr(torch.cuda, 'amp'):
    _original_autocast = torch.cuda.amp.autocast

    def autocast_replacement(*args, **kwargs):
        default_device = get_default_device()
        if default_device != 'cuda':
            log_warning("torch.cuda.amp.autocast called on a non-CUDA device; behavior may be unexpected.", "torch.cuda.amp.autocast")
        return _original_autocast(*args, **kwargs)

    torch.cuda.amp.autocast = autocast_replacement

    if hasattr(torch.cuda.amp, 'GradScaler'):
        _OriginalGradScaler = torch.cuda.amp.GradScaler

        class GradScalerReplacement(_OriginalGradScaler):
            def __init__(self, *args, **kwargs):
                if get_default_device() != 'cuda':
                    log_warning("torch.cuda.amp.GradScaler instantiated on a non-CUDA device; behavior may be unexpected.", "torch.cuda.amp.GradScaler")
                super().__init__(*args, **kwargs)
        torch.cuda.amp.GradScaler = GradScalerReplacement

# --- TorchDevice Class with Patched CUDA Functions ---

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
                log_info(f"Creating torch.device('{device_str}')", "torch.device")
                self.device = self.__class__._original_torch_device(device_str)
            else:
                self.device = self.__class__._original_torch_device(device_type)

    def __repr__(self):
        return repr(self.device)

    def __str__(self):
        return str(self.device)

    @classmethod
    def _detect_default_device(cls):
        if torch.backends.mps.is_available():
            log_info("MPS (Apple Silicon) detected as default device", "TorchDevice")
            cls._default_device = 'mps'
        elif cls._original_torch_cuda_is_available():
            log_info("CUDA detected as default device", "TorchDevice")
            cls._default_device = 'cuda'
        else:
            log_info("No GPU detected, using CPU as default device", "TorchDevice")
            cls._default_device = 'cpu'
        log_info(f"Default device set to: {cls._default_device}", "TorchDevice")

    @classmethod
    def _redirect_device_type(cls, device_type):
        if device_type.startswith('cuda'):
            if cls._default_device == 'cuda':
                return 'cuda'
            elif cls._default_device == 'mps':
                log_warning("CUDA device requested but not available. Redirecting to MPS.", "torch.device")
                return 'mps'
            else:
                log_warning("CUDA device requested but not available. Redirecting to CPU.", "torch.device")
                return 'cpu'
        elif device_type.startswith('mps'):
            if cls._default_device == 'mps':
                return 'mps'
            elif cls._default_device == 'cuda':
                log_warning("MPS device requested but not available. Redirecting to CUDA.", "torch.device")
                return 'cuda'
            else:
                log_warning("MPS device requested but not available. Redirecting to CPU.", "torch.device")
                return 'cpu'
        else:
            return device_type

    def __getattr__(self, attr):
        return getattr(self.device, attr)

    @classmethod
    def torch_device_replacement(cls, device_type=None, device_index=None):
        with cls._lock:
            if cls._default_device is None:
                cls._detect_default_device()
            if device_type is None:
                device_type = cls._default_device
                device_index = 0 if device_index is None else device_index
                device_str = f"{device_type}:{device_index}"
                log_info(f"Creating torch.device('{device_str}')", "torch.device")
                return cls._original_torch_device(device_str)
            if isinstance(device_type, str):
                if ':' in device_type:
                    device_type, index = device_type.split(':')
                    device_index = int(index)
                else:
                    device_index = 0 if device_index is None else device_index
                redirected_device_type = cls._redirect_device_type(device_type)
                device_str = f"{redirected_device_type}:{device_index}"
                if redirected_device_type != device_type:
                    log_info(f"Redirecting torch.device('{device_type}:{device_index}') to torch.device('{device_str}')", "torch.device")
                else:
                    log_info(f"Creating torch.device('{device_str}')", "torch.device")
                return cls._original_torch_device(device_str)
            else:
                return cls._original_torch_device(device_type)

    @classmethod
    def mock_cuda_is_available(cls):
        if cls._default_device in ['cuda', 'mps']:
            log_info("CUDA is available.", "torch.cuda.is_available")
            return True
        else:
            log_warning("CUDA is not available.", "torch.cuda.is_available")
            return False

    @classmethod
    def mock_cuda_device_count(cls):
        if cls._default_device == 'cuda':
            count = cls._original_torch_cuda_device_count()
            log_info(f"CUDA device count: {count}", "torch.cuda.device_count")
            return count
        elif cls._default_device == 'mps':
            log_info("Returning device count as 1 for MPS.", "torch.cuda.device_count")
            return 1
        else:
            log_warning("CUDA device count requested but no GPU is available. Returning 0.", "torch.cuda.device_count")
            return 0

    @classmethod
    def mock_cuda_get_device_properties(cls, device):
        if cls._default_device == 'cuda':
            props = cls._original_torch_cuda_get_device_properties(device)
            log_info(f"CUDA device properties for device {device}: {props}", "torch.cuda.get_device_properties")
            return props
        elif cls._default_device == 'mps':
            log_info("Returning MPS device properties.", "torch.cuda.get_device_properties")
            class MPSDeviceProperties:
                name = 'Apple MPS'
                total_memory = psutil.virtual_memory().total
                def __str__(self):
                    return f'MPSDeviceProperties(name={self.name}, total_memory={self.total_memory})'
            return MPSDeviceProperties()
        else:
            log_error("No GPU device available.", "torch.cuda.get_device_properties")
            raise RuntimeError("No GPU device available")

    @classmethod
    def mock_cuda_memory_allocated(cls, device=None):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    @classmethod
    def mock_cuda_memory_reserved(cls, device=None):
        return psutil.virtual_memory().total

    @classmethod
    def mock_cuda_max_memory_allocated(cls, device=None):
        return cls.mock_cuda_memory_allocated(device)

    @classmethod
    def mock_cuda_max_memory_reserved(cls, device=None):
        return cls.mock_cuda_memory_reserved(device)

    @classmethod
    def mock_cuda_empty_cache(cls):
        if cls._default_device == 'cuda':
            log_info("Clearing CUDA cache.", "torch.cuda.empty_cache")
            cls._original_torch_cuda_empty_cache()
        elif cls._default_device == 'mps':
            log_info("Clearing MPS cache.", "torch.cuda.empty_cache")
            torch.mps.empty_cache()
        else:
            log_warning("No GPU cache to clear.", "torch.cuda.empty_cache")

    @classmethod
    def mock_cuda_synchronize(cls, device=None):
        if cls._default_device == 'cuda':
            log_info("Synchronizing CUDA.", "torch.cuda.synchronize")
            cls._original_torch_cuda_synchronize(device)
        elif cls._default_device == 'mps':
            log_info("Synchronizing MPS.", "torch.cuda.synchronize")
            torch.mps.synchronize()
        else:
            log_warning("No GPU to synchronize.", "torch.cuda.synchronize")

    @classmethod
    def mock_cuda_current_device(cls):
        if cls._default_device == 'cuda':
            current_device = cls._original_torch_cuda_current_device()
            log_info(f"Current CUDA device: {current_device}", "torch.cuda.current_device")
            return current_device
        elif cls._default_device == 'mps':
            log_info("Returning current MPS device (0).", "torch.cuda.current_device")
            return 0
        else:
            log_warning("No GPU available. Returning -1.", "torch.cuda.current_device")
            return -1

    @classmethod
    def mock_cuda_set_device(cls, device):
        if cls._default_device == 'cuda':
            log_info(f"Setting CUDA device to {device}", "torch.cuda.set_device")
            cls._original_torch_cuda_set_device(device)
        elif cls._default_device == 'mps':
            log_warning("MPS does not support setting device.", "torch.cuda.set_device")
        else:
            log_warning("No GPU available to set device.", "torch.cuda.set_device")

    @classmethod
    def mock_cuda_get_device_name(cls, device=None):
        if cls._default_device == 'cuda':
            name = cls._original_torch_cuda_get_device_name(device)
            log_info(f"CUDA device name: {name}", "torch.cuda.get_device_name")
            return name
        elif cls._default_device == 'mps':
            log_info("Returning 'Apple MPS' as device name.", "torch.cuda.get_device_name")
            return 'Apple MPS'
        else:
            log_warning("No GPU available to get device name.", "torch.cuda.get_device_name")
            return 'CPU'

    @classmethod
    def mock_cuda_get_device_capability(cls, device=None):
        if cls._default_device == 'cuda':
            cap = cls._original_torch_cuda_get_device_capability(device)
            log_info(f"CUDA device capability: {cap}", "torch.cuda.get_device_capability")
            return cap
        elif cls._default_device == 'mps':
            log_info("Returning (0, 0) for MPS device capability.", "torch.cuda.get_device_capability")
            return (0, 0)
        else:
            log_warning("No GPU available to get device capability.", "torch.cuda.get_device_capability")
            return (0, 0)

    @classmethod
    def mock_cuda_memory_stats(cls, device=None):
        stats = {
            'active.all.current': cls.mock_cuda_memory_allocated(device),
            'reserved_bytes.all.current': cls.mock_cuda_memory_reserved(device),
        }
        log_info(f"Memory stats: {stats}", "torch.cuda.memory_stats")
        return stats

    @classmethod
    def mock_cuda_memory_snapshot(cls):
        log_info("Returning empty memory snapshot.", "torch.cuda.memory_snapshot")
        return []

    @classmethod
    def mock_cuda_memory_summary(cls, device=None, abbreviated=False):
        log_info("Generating memory summary.", "torch.cuda.memory_summary")
        summary = f"Memory Allocated: {cls.mock_cuda_memory_allocated(device)} bytes\n"
        summary += f"Memory Reserved: {cls.mock_cuda_memory_reserved(device)} bytes\n"
        return summary

    @classmethod
    def mock_cuda_is_initialized(cls):
        if cls._default_device in ['cuda', 'mps']:
            log_info("CUDA is initialized.", "torch.cuda.is_initialized")
            return True
        else:
            log_warning("CUDA is not initialized.", "torch.cuda.is_initialized")
            return False

    @classmethod
    def mock_cuda_get_arch_list(cls):
        if cls._default_device == 'cuda':
            arch_list = cls._original_torch_cuda_get_arch_list()
            log_info(f"CUDA arch list: {arch_list}", "torch.cuda.get_arch_list")
            return arch_list
        elif cls._default_device == 'mps':
            log_info("Returning ['mps'] as arch list.", "torch.cuda.get_arch_list")
            return ['mps']
        else:
            log_warning("No GPU available. Returning empty arch list.", "torch.cuda.get_arch_list")
            return []

    @classmethod
    def mock_cuda_is_built(cls):
        if cls._default_device == 'cuda':
            log_info("CUDA backend is built.", "torch.backends.cuda.is_built")
            return True
        elif cls._default_device == 'mps':
            log_warning("CUDA backend is not built, but MPS backend is built. Reporting as built.", "torch.backends.cuda.is_built")
            return True
        else:
            log_warning("Neither CUDA nor MPS backend is built.", "torch.backends.cuda.is_built")
            return False

    @classmethod
    def mock_cuda_device_context(cls, device=None):
        class DeviceContextManager:
            def __init__(self, device):
                self.device = device
            def __enter__(self):
                cls.mock_cuda_set_device(self.device)
            def __exit__(self, exc_type, exc_value, traceback):
                pass
        return DeviceContextManager(device)

    @classmethod
    def mock_cuda_reset_peak_memory_stats(cls):
        if cls._default_device in ['cuda', 'mps']:
            log_info("Resetting peak memory stats.", "torch.cuda.reset_peak_memory_stats")
        else:
            log_warning("No GPU available to reset peak memory stats.", "torch.cuda.reset_peak_memory_stats")

    @classmethod
    def mock_cuda_ipc_collect(cls):
        if cls._default_device in ['cuda', 'mps']:
            log_info("Collecting IPC memory.", "torch.cuda.ipc_collect")
        else:
            log_warning("No GPU available to collect IPC memory.", "torch.cuda.ipc_collect")

    @classmethod
    def mock_cuda_stream(cls, *args, **kwargs):
        log_warning("CUDA streams are not supported on this device. Ignoring.", "torch.cuda.stream")

    @classmethod
    def mock_cuda_stream_class(cls, *args, **kwargs):
        log_warning("CUDA Stream class is not supported on this device.", "torch.cuda.Stream")
        class MockStream:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_value, traceback): pass
        return MockStream()

    @classmethod
    def mock_cuda_event(cls, *args, **kwargs):
        log_warning("CUDA Event is not supported on this device.", "torch.cuda.Event")
        class MockEvent:
            def __init__(self, *args, **kwargs): pass
            def record(self, *args, **kwargs): pass
            def wait(self, *args, **kwargs): pass
            def query(self): return True
        return MockEvent()

    @staticmethod
    def mock_cuda_function_stub(*args, **kwargs):
        log_warning("Unsupported CUDA function called. Ignoring.", "torch.cuda")

    @classmethod
    def apply_patches(cls):
        # Replace torch.device with our TorchDevice replacement
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
        for func_name in unsupported_functions:
            if hasattr(torch.cuda, func_name):
                setattr(torch.cuda, func_name, cls.mock_cuda_function_stub)

# Apply patches when the module is imported
TorchDevice.apply_patches()