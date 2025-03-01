import inspect
import logging
import os
import sys
import threading
import torch
import psutil
import types
import time

# Get the absolute path of the current module
_CURRENT_MODULE = os.path.abspath(__file__)

# --- Capture Original torch.device Type ---
# This must be done before any patches to torch.device occur.
_ORIGINAL_TORCH_DEVICE_TYPE = torch.device("cpu").__class__

# Set up logging
logger = logging.getLogger('TorchDevice')
logger.setLevel(logging.INFO)

# Create a formatter that includes the file name
formatter = logging.Formatter('[%(caller_filename)s]: %(message)s')

# Create a console handler and set the formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

# Message deduplication cache
_message_cache = {}
_cache_timeout = 0.5  # seconds

def get_caller_info():
    """
    Walk the stack and return info from the first frame that is not:
      - from this module,
      - from internal frameworks (e.g. unittest, inspect), or
      - a function known to be part of test harness setup (like setUp/tearDown).
    """
    program_name = os.path.basename(sys.argv[0])
    excluded_functions = {"setUp", "tearDown", "run"}  # add more if needed

    # Get the full stack trace
    stack = inspect.stack()

    # Skip the first few frames which are the logging functions
    for frame_record in stack[3:]:
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

        # Get the source line of code that made the call
        source_line = ""
        try:
            source_lines, _ = inspect.getsourcelines(frame_record.frame)
            line_in_function = frame_record.lineno - frame_record.frame.f_code.co_firstlineno
            if line_in_function >= 0 and line_in_function < len(source_lines):
                source_line = source_lines[line_in_function].strip()
            else:
                source_line = "Source line unavailable"
        except Exception:
            source_line = "Source line unavailable"

        # Get the arguments of the call if possible
        call_args = {}
        try:
            for arg_name, arg_value in frame_record.frame.f_locals.items():
                if arg_name != 'self' and not arg_name.startswith('__'):
                    if isinstance(arg_value, (str, int, float, bool, type(None))):
                        call_args[arg_name] = arg_value
                    else:
                        call_args[arg_name] = str(type(arg_value))
        except Exception:
            call_args = {"error": "Could not extract arguments"}

        return {
            'program_name': program_name,
            'module_name': mod_name if mod_name else 'UnknownModule',
            'caller_filename': os.path.basename(abs_filename),
            'class_name': (frame_record.frame.f_locals.get('self').__class__.__name__
                           if 'self' in frame_record.frame.f_locals else 'N/A'),
            'caller_lineno': frame_record.lineno,
            'caller_func_name': frame_record.function,
            'source_line': source_line,
            'call_args': str(call_args)
        }
    # Fallback if no candidate is found
    return {
        'program_name': program_name,
        'module_name': 'UnknownModule',
        'caller_filename': 'UnknownFile',
        'class_name': 'N/A',
        'caller_lineno': 0,
        'caller_func_name': 'UnknownFunction',
        'source_line': 'Unknown',
        'call_args': '{}'
    }

def log_message(level, message, torch_function=None, device_type=None):
    info = get_caller_info()

    # Get the caller's function and line number
    caller_func = info.get('caller_func_name', 'unknown')
    caller_line = info.get('caller_lineno', 0)

    # Format the device information
    device_info = ""
    if device_type:
        device_info = "[%s]" % device_type
    elif "cuda" in message.lower():
        device_info = "[CUDA→MPS]"
    elif "mps" in message.lower():
        device_info = "[MPS]"
    elif "cpu" in message.lower():
        device_info = "[CPU]"

    # Format the torch function information
    function_info = ""
    if torch_function:
        function_info = "[%s]" % torch_function

    # Create a cleaner log message format
    formatted_message = "GPU REDIRECT in \"%s\" line: %d %s %s %s" % (
        caller_func, caller_line, function_info, device_info, message
    )

    # Check if the message is already in the cache
    cache_key = (caller_func, caller_line, formatted_message)
    if cache_key in _message_cache and time.time() - _message_cache[cache_key] < _cache_timeout:
        return

    # Log the message
    logger.log(level, formatted_message, extra=info)

    # Add the message to the cache
    _message_cache[cache_key] = time.time()

def log_info(message, torch_function=None, device_type=None):
    log_message(logging.INFO, message, torch_function, device_type)

def log_warning(message, torch_function=None, device_type=None):
    log_message(logging.WARNING, message, torch_function, device_type)

def log_error(message, torch_function=None, device_type=None):
    log_message(logging.ERROR, message, torch_function, device_type)

# --- PyTorch Function Replacements ---

_in_torch_load = False
_original_torch_load = torch.load
_original_tensor_cuda = torch.Tensor.cuda
_original_module_cuda = torch.nn.Module.cuda
_original_tensor_to = torch.Tensor.to
_original_module_to = torch.nn.Module.to

def tensor_cuda_replacement(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
    default_device = TorchDevice.get_default_device()
    log_info("tensor.cuda() called with device=%s" % device, "tensor.cuda", device_type=default_device)
    if default_device == 'mps':
        log_info("Redirecting tensor.cuda() to tensor.to('mps')", "tensor.cuda", device_type=default_device)
        return self.to('mps', non_blocking=non_blocking, memory_format=memory_format)
    return _original_tensor_cuda(self, device, non_blocking, memory_format)

def module_cuda_replacement(self, device=None):
    default_device = TorchDevice.get_default_device()
    log_info("nn.Module.cuda() called with device=%s" % device, "nn.Module.cuda", device_type=default_device)
    if default_device == 'mps':
        log_info("Redirecting nn.Module.cuda() to nn.Module.to('mps')", "nn.Module.cuda", device_type=default_device)
        return self.to('mps')
    return _original_module_cuda(self, device)

# Intercept .to() calls for both tensors and modules
def tensor_to_replacement(self, *args, **kwargs):
    # Original implementation of tensor.to
    default_device = TorchDevice.get_default_device()

    # Check if a device is specified in args or kwargs
    device_arg = None
    if args and isinstance(args[0], (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
        device_arg = args[0]
    elif 'device' in kwargs:
        device_arg = kwargs['device']

    # If a device is specified, log a warning if it doesn't match the default
    if device_arg is not None:
        if isinstance(device_arg, _ORIGINAL_TORCH_DEVICE_TYPE):
            target_device = str(device_arg.type)
        else:
            target_device = str(device_arg)
        if target_device != default_device:
            log_warning("tensor.to() called with device %s which does not match the default device %s." % 
                       (device_arg, default_device), "tensor.to", device_type=default_device)
    return _original_tensor_to(self, *args, **kwargs)

def module_to_replacement(self, *args, **kwargs):
    # Original implementation of module.to
    default_device = TorchDevice.get_default_device()

    # Check if a device is specified in args or kwargs
    device_arg = None
    if args:
        candidate = args[0]
        if isinstance(candidate, (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
            device_arg = candidate
    elif 'device' in kwargs:
        device_arg = kwargs['device']

    # If a device is specified, log a warning if it doesn't match the default
    if device_arg is not None:
        if isinstance(device_arg, _ORIGINAL_TORCH_DEVICE_TYPE):
            target_device = str(device_arg.type)
        else:
            target_device = str(device_arg)
        if target_device != default_device:
            log_warning("module.to() called with device %s which does not match the default device %s." % 
                       (device_arg, default_device), "module.to", device_type=default_device)
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
        default_device = TorchDevice.get_default_device()
        log_info("torch.load called with args: %s, kwargs: %s" % (args, kwargs), "torch.load", device_type=default_device)
        if 'map_location' in kwargs:
            if kwargs['map_location'] == 'cpu' or (isinstance(kwargs['map_location'], str) and kwargs['map_location'] != default_device):
                log_info("Replacing map_location=%s with %s" % (kwargs['map_location'], default_device), "torch.load", device_type=default_device)
                kwargs['map_location'] = default_device
        else:
            log_info("Adding map_location=%s" % default_device, "torch.load", device_type=default_device)
            kwargs['map_location'] = default_device
        return _original_torch_load(*args, **kwargs)
    finally:
        _in_torch_load = False

torch.load = torch_load_replacement

# --- TorchDevice Class with Patched CUDA Functions ---

class TorchDevice:
    # Class variables for caching and locking
    _default_device = None
    _lock = threading.Lock()

    # Store original PyTorch functions
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

    @classmethod
    def get_default_device(cls):
        """
        Detect and return the default device based on available hardware.
        Uses a cached value to avoid redundant detection.
        """
        with cls._lock:
            if cls._default_device is None:
                # Detect the default device
                if torch.backends.mps.is_available():
                    cls._default_device = 'mps'
                    log_info("MPS (Apple Silicon) detected and set as default device", "TorchDevice", device_type='mps')
                elif cls._original_torch_cuda_is_available():
                    cls._default_device = 'cuda'
                    log_info("CUDA detected and set as default device", "TorchDevice", device_type='cuda')
                else:
                    cls._default_device = 'cpu'
                    log_info("No GPU detected, CPU set as default device", "TorchDevice", device_type='cpu')

            return cls._default_device

    @classmethod
    def redirect_device_type(cls, device_type):
        """
        Redirect device type if the requested device is not available.
        Returns the appropriate device type based on availability.
        """
        # Ensure default device is detected
        default_device = cls.get_default_device()

        # Handle CUDA redirection
        if device_type == 'cuda' and default_device != 'cuda':
            if default_device == 'mps':
                log_warning("CUDA device requested but not available. Redirecting to MPS.", "torch.device", device_type='mps')
                return 'mps'
            else:
                log_warning("CUDA device requested but not available. Redirecting to CPU.", "torch.device", device_type='cpu')
                return 'cpu'

        # Handle MPS redirection
        elif device_type == 'mps' and default_device != 'mps':
            if default_device == 'cuda':
                log_warning("MPS device requested but not available. Redirecting to CUDA.", "torch.device", device_type='cuda')
                return 'cuda'
            else:
                log_warning("MPS device requested but not available. Redirecting to CPU.", "torch.device", device_type='cpu')
                return 'cpu'

        # No redirection needed
        return device_type

    @classmethod
    def create_device(cls, device_type=None, device_index=None):
        """
        Create a device with the specified type and index.
        Handles device redirection and formatting.
        """
        # Ensure default device is detected
        if device_type is None:
            device_type = cls.get_default_device()

        # Handle string device types
        if isinstance(device_type, str):
            # Parse device string if it contains an index
            if ':' in device_type:
                device_type, index = device_type.split(':')
                device_index = int(index)
            else:
                device_index = 0 if device_index is None else device_index

            # Redirect device type if needed
            redirected_device_type = cls.redirect_device_type(device_type)
            device_str = f"{redirected_device_type}:{device_index}"

            # Log redirection if it occurred
            if redirected_device_type != device_type:
                log_info("Redirecting torch.device('%s') to torch.device('%s')" % (
                    f"{device_type}:{device_index}", device_str), 
                    "torch.device", device_type=redirected_device_type)

            return cls._original_torch_device(device_str)
        else:
            # Handle non-string device types
            return cls._original_torch_device(device_type)

    def __init__(self, device_type: str = None, device_index: int = None):
        """
        Initialize a TorchDevice instance.
        """
        # Create the device using the consolidated method
        self.device = self.__class__.create_device(device_type, device_index)

    def __repr__(self):
        return repr(self.device)

    def __str__(self):
        return str(self.device)

    def __getattr__(self, attr):
        return getattr(self.device, attr)

    @classmethod
    def mock_cuda_is_available(cls):
        if cls.get_default_device() in ['cuda', 'mps']:
            log_info("CUDA is available.", "torch.cuda.is_available", device_type=cls.get_default_device())
            return True
        else:
            log_warning("CUDA is not available.", "torch.cuda.is_available", device_type='cpu')
            return False

    @classmethod
    def mock_cuda_device_count(cls):
        if cls.get_default_device() == 'cuda':
            count = cls._original_torch_cuda_device_count()
            log_info("CUDA device count: %s" % count, "torch.cuda.device_count", device_type='cuda')
            return count
        elif cls.get_default_device() == 'mps':
            log_info("Returning device count as 1 for MPS.", "torch.cuda.device_count", device_type='mps')
            return 1
        else:
            log_warning("CUDA device count requested but no GPU is available. Returning 0.", "torch.cuda.device_count", device_type='cpu')
            return 0

    @classmethod
    def mock_cuda_get_device_properties(cls, device):
        if cls.get_default_device() == 'cuda':
            props = cls._original_torch_cuda_get_device_properties(device)
            log_info("CUDA device properties for device %s: %s" % (device, props), "torch.cuda.get_device_properties", device_type='cuda')
            return props
        elif cls.get_default_device() == 'mps':
            log_info("Returning MPS device properties.", "torch.cuda.get_device_properties", device_type='mps')
            class MPSDeviceProperties:
                name = 'Apple MPS'
                total_memory = psutil.virtual_memory().total
                def __str__(self):
                    return f'MPSDeviceProperties(name={self.name}, total_memory={self.total_memory})'
            return MPSDeviceProperties()
        else:
            log_error("No GPU device available.", "torch.cuda.get_device_properties", device_type='cpu')
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
        if cls.get_default_device() == 'cuda':
            log_info("Clearing CUDA cache.", "torch.cuda.empty_cache", device_type='cuda')
            cls._original_torch_cuda_empty_cache()
        elif cls.get_default_device() == 'mps':
            log_info("Clearing MPS cache.", "torch.cuda.empty_cache", device_type='mps')
            torch.mps.empty_cache()
        else:
            log_warning("No GPU cache to clear.", "torch.cuda.empty_cache", device_type='cpu')

    @classmethod
    def mock_cuda_synchronize(cls, device=None):
        if cls.get_default_device() == 'cuda':
            log_info("Synchronizing CUDA.", "torch.cuda.synchronize", device_type='cuda')
            cls._original_torch_cuda_synchronize(device)
        elif cls.get_default_device() == 'mps':
            log_info("Synchronizing MPS.", "torch.cuda.synchronize", device_type='mps')
            torch.mps.synchronize()
        else:
            log_warning("No GPU to synchronize.", "torch.cuda.synchronize", device_type='cpu')

    @classmethod
    def mock_cuda_current_device(cls):
        if cls.get_default_device() == 'cuda':
            current_device = cls._original_torch_cuda_current_device()
            log_info("Current CUDA device: %s" % current_device, "torch.cuda.current_device", device_type='cuda')
            return current_device
        elif cls.get_default_device() == 'mps':
            log_info("Returning current MPS device (0).", "torch.cuda.current_device", device_type='mps')
            return 0
        else:
            log_warning("No GPU available. Returning -1.", "torch.cuda.current_device", device_type='cpu')
            return -1

    @classmethod
    def mock_cuda_set_device(cls, device):
        if cls.get_default_device() == 'cuda':
            log_info("Setting CUDA device to %s" % device, "torch.cuda.set_device", device_type='cuda')
            cls._original_torch_cuda_set_device(device)
        elif cls.get_default_device() == 'mps':
            log_warning("MPS does not support setting device.", "torch.cuda.set_device", device_type='mps')
        else:
            log_warning("No GPU available to set device.", "torch.cuda.set_device", device_type='cpu')

    @classmethod
    def mock_cuda_get_device_name(cls, device=None):
        if cls.get_default_device() == 'cuda':
            name = cls._original_torch_cuda_get_device_name(device)
            log_info("CUDA device name: %s" % name, "torch.cuda.get_device_name", device_type='cuda')
            return name
        elif cls.get_default_device() == 'mps':
            log_info("Returning 'Apple MPS' as device name.", "torch.cuda.get_device_name", device_type='mps')
            return 'Apple MPS'
        else:
            log_warning("No GPU available to get device name.", "torch.cuda.get_device_name", device_type='cpu')
            return 'CPU'

    @classmethod
    def mock_cuda_get_device_capability(cls, device=None):
        if cls.get_default_device() == 'cuda':
            cap = cls._original_torch_cuda_get_device_capability(device)
            log_info("CUDA device capability: %s" % cap, "torch.cuda.get_device_capability", device_type='cuda')
            return cap
        elif cls.get_default_device() == 'mps':
            log_info("Returning (0, 0) for MPS device capability.", "torch.cuda.get_device_capability", device_type='mps')
            return (0, 0)
        else:
            log_warning("No GPU available to get device capability.", "torch.cuda.get_device_capability", device_type='cpu')
            return (0, 0)

    @classmethod
    def mock_cuda_memory_stats(cls, device=None):
        stats = {
            'active.all.current': cls.mock_cuda_memory_allocated(device),
            'reserved_bytes.all.current': cls.mock_cuda_memory_reserved(device),
        }
        log_info("Memory stats: %s" % stats, "torch.cuda.memory_stats", device_type=cls.get_default_device())
        return stats

    @classmethod
    def mock_cuda_memory_snapshot(cls):
        log_info("Returning empty memory snapshot.", "torch.cuda.memory_snapshot", device_type=cls.get_default_device())
        return []

    @classmethod
    def mock_cuda_memory_summary(cls, device=None, abbreviated=False):
        log_info("Generating memory summary.", "torch.cuda.memory_summary", device_type=cls.get_default_device())
        summary = "Memory Allocated: %s bytes\n" % cls.mock_cuda_memory_allocated(device)
        summary += "Memory Reserved: %s bytes\n" % cls.mock_cuda_memory_reserved(device)
        return summary

    @classmethod
    def mock_cuda_is_initialized(cls):
        if cls.get_default_device() in ['cuda', 'mps']:
            log_info("CUDA is initialized.", "torch.cuda.is_initialized", device_type=cls.get_default_device())
            return True
        else:
            log_warning("CUDA is not initialized.", "torch.cuda.is_initialized", device_type='cpu')
            return False

    @classmethod
    def mock_cuda_get_arch_list(cls):
        if cls.get_default_device() == 'cuda':
            arch_list = cls._original_torch_cuda_get_arch_list()
            log_info("CUDA arch list: %s" % arch_list, "torch.cuda.get_arch_list", device_type='cuda')
            return arch_list
        elif cls.get_default_device() == 'mps':
            log_info("Returning ['mps'] as arch list.", "torch.cuda.get_arch_list", device_type='mps')
            return ['mps']
        else:
            log_warning("No GPU available. Returning empty arch list.", "torch.cuda.get_arch_list", device_type='cpu')
            return []

    @classmethod
    def mock_cuda_is_built(cls):
        if cls.get_default_device() == 'cuda':
            log_info("CUDA backend is built.", "torch.backends.cuda.is_built", device_type='cuda')
            return True
        elif cls.get_default_device() == 'mps':
            log_warning("CUDA backend is not built, but MPS backend is built. Reporting as built.", "torch.backends.cuda.is_built", device_type='mps')
            return True
        else:
            log_warning("Neither CUDA nor MPS backend is built.", "torch.backends.cuda.is_built", device_type='cpu')
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
        if cls.get_default_device() in ['cuda', 'mps']:
            log_info("Resetting peak memory stats.", "torch.cuda.reset_peak_memory_stats", device_type=cls.get_default_device())
        else:
            log_warning("No GPU available to reset peak memory stats.", "torch.cuda.reset_peak_memory_stats", device_type='cpu')

    @classmethod
    def mock_cuda_ipc_collect(cls):
        if cls.get_default_device() in ['cuda', 'mps']:
            log_info("Collecting IPC memory.", "torch.cuda.ipc_collect", device_type=cls.get_default_device())
        else:
            log_warning("No GPU available to collect IPC memory.", "torch.cuda.ipc_collect", device_type='cpu')

    @classmethod
    def mock_cuda_stream(cls, stream=None):
        """Mock for torch.cuda.stream function."""
        from TorchDevice.TorchDevice import log_info, log_warning
        
        if cls.get_default_device() == 'cuda':
            log_info("Creating CUDA stream (mocked).", "torch.cuda.stream", device_type='cuda')
            # Return a simple object that mimics a CUDA stream
            class MockStream:
                def __init__(self):
                    self._stream = stream

                def query(self):
                    log_info("Stream query called.", "torch.cuda.stream.query", device_type='cuda')
                    return True

                def synchronize(self):
                    log_info("Stream synchronize called.", "torch.cuda.stream.synchronize", device_type='cuda')

                def record_event(self, event=None):
                    log_info("Stream record_event called.", "torch.cuda.stream.record_event", device_type='cuda')
                    if event is None:
                        # Create a mock event if none provided
                        return cls.mock_cuda_event()
                    return event

                def __eq__(self, other):
                    return isinstance(other, MockStream) and self._stream == other._stream
                
                def __enter__(self):
                    log_info("Stream context manager __enter__ called.", "torch.cuda.stream.__enter__", device_type='cuda')
                    return self
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    log_info("Stream context manager __exit__ called.", "torch.cuda.stream.__exit__", device_type='cuda')
                    return False

            return MockStream()
        elif cls.get_default_device() == 'mps':
            log_info("Creating MPS stream (mocked).", "torch.cuda.stream", device_type='mps')
            
            # Create a complete mock implementation for MPS
            class MockMPSStream:
                def __init__(self):
                    self._stream = stream
                    self._device = 'mps'
            
                def query(self):
                    log_info("Stream query called.", "torch.cuda.stream.query -> MPS", device_type='mps')
                    return True

                def synchronize(self):
                    log_info("Stream synchronize called.", "torch.cuda.stream.synchronize -> MPS", device_type='mps')

                def record_event(self, event=None):
                    log_info("Stream record_event called.", "torch.cuda.stream.record_event -> MPS", device_type='mps')
                    if event is None:
                        # Create a mock event if none provided
                        return cls.mock_cuda_event()
                    return event
                
                def wait_event(self, event):
                    log_info("Stream wait_event called.", "torch.cuda.stream.wait_event -> MPS", device_type='mps')
                    return self
            
                def wait_stream(self, stream):
                    log_info("Stream wait_stream called.", "torch.cuda.stream.wait_stream -> MPS", device_type='mps')
                    return self

                def __eq__(self, other):
                    return isinstance(other, MockMPSStream) and self._stream == other._stream
                
                def __enter__(self):
                    log_info("Stream context manager __enter__ called.", "torch.cuda.stream.__enter__ -> MPS", device_type='mps')
                    return self
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    log_info("Stream context manager __exit__ called.", "torch.cuda.stream.__exit__ -> MPS", device_type='mps')
                    return False

            return MockMPSStream()
        else:
            log_warning("No GPU available to create stream.", "torch.cuda.stream", device_type='cpu')
            return None

    @classmethod
    def mock_cuda_current_stream(cls):
        """Mock for torch.cuda.current_stream function."""
        if cls.get_default_device() == 'cuda':
            log_info("Getting current CUDA stream (mocked).", "torch.cuda.current_stream", device_type='cuda')
            return cls.mock_cuda_stream()
        elif cls.get_default_device() == 'mps':
            # If MPS is available, try to use the real MPS current_stream function
            try:
                import torch.mps
                if hasattr(torch.mps, 'current_stream'):
                    log_info("Getting current MPS stream.", "torch.cuda.current_stream -> torch.mps.current_stream", device_type='mps')
                    return torch.mps.current_stream()
                else:
                    # Fall back to mock implementation
                    log_info("Getting current MPS stream (mocked).", "torch.cuda.current_stream", device_type='mps')
                    return cls.mock_cuda_stream()
            except ImportError:
                log_info("Getting current MPS stream (mocked).", "torch.cuda.current_stream", device_type='mps')
                return cls.mock_cuda_stream()
        else:
            log_warning("No GPU available to get current stream.", "torch.cuda.current_stream", device_type='cpu')
            return None

    @classmethod
    def mock_cuda_default_stream(cls):
        """Mock for torch.cuda.default_stream function."""
        if cls.get_default_device() == 'cuda':
            log_info("Getting default CUDA stream (mocked).", "torch.cuda.default_stream", device_type='cuda')
            return cls.mock_cuda_stream()
        elif cls.get_default_device() == 'mps':
            # If MPS is available, try to use the real MPS default_stream function
            try:
                import torch.mps
                if hasattr(torch.mps, 'default_stream'):
                    log_info("Getting default MPS stream.", "torch.cuda.default_stream -> torch.mps.default_stream", device_type='mps')
                    return torch.mps.default_stream()
                else:
                    # Fall back to mock implementation
                    log_info("Getting default MPS stream (mocked).", "torch.cuda.default_stream", device_type='mps')
                    return cls.mock_cuda_stream()
            except ImportError:
                log_info("Getting default MPS stream (mocked).", "torch.cuda.default_stream", device_type='mps')
                return cls.mock_cuda_stream()
        else:
            log_warning("No GPU available to get default stream.", "torch.cuda.default_stream", device_type='cpu')
            return None

    @classmethod
    def mock_cuda_event(cls, enable_timing=False, blocking=False, interprocess=False):
        """
        Mock the torch.cuda.Event class.
        """
        log_info("Creating MPS event.", "torch.cuda.Event -> torch.mps.event.Event", device_type='mps')

        try:
            # Try to use MPS event if available
            if hasattr(torch.mps, 'event') and hasattr(torch.mps.event, 'Event'):
                mps_event = torch.mps.event.Event()

                # Add a wrapper for the record method to handle stream parameter
                original_record = mps_event.record
                def record_wrapper(stream=None):
                    log_info("Event record called, ignoring stream parameter for MPS compatibility.", 
                             "torch.mps.event.Event.record", device_type='mps')
                    return original_record()

                mps_event.record = record_wrapper

                return mps_event
        except (ImportError, AttributeError) as e:
            log_warning(f"MPS event not available: {e}", "torch.cuda.Event", device_type='mps')

        # Fallback to mock implementation
        class MockEvent:
            def __init__(self):
                self.recorded = False
                self.enable_timing = enable_timing
                self.blocking = blocking
                self.interprocess = interprocess
                self.start_time = None
                self.end_time = None

            def record(self, stream=None):
                log_info("Event record called.", "torch.cuda.Event.record -> MPS", device_type='mps')
                self.recorded = True
                self.start_time = time.time()
                return None

            def synchronize(self):
                log_info("Event synchronize called.", "torch.cuda.Event.synchronize -> MPS", device_type='mps')
                return None

            def wait(self, stream=None):
                log_info("Event wait called.", "torch.cuda.Event.wait -> MPS", device_type='mps')
                return None

            def query(self):
                log_info("Event query called.", "torch.cuda.Event.query -> MPS", device_type='mps')
                return True

            def elapsed_time(self, end_event):
                log_info("Event elapsed_time called.", "torch.cuda.Event.elapsed_time -> MPS", device_type='mps')
                if not self.enable_timing:
                    raise RuntimeError("Events were created without timing enabled")

                if not self.recorded or not end_event.recorded:
                    raise RuntimeError("Events must be recorded before elapsed_time can be computed")

                # If we have actual timestamps, use them
                if self.start_time and end_event.end_time:
                    return (end_event.end_time - self.start_time) * 1000.0  # Convert to ms

                # Otherwise return a mock value
                return 0.5  # Return a small mock value in milliseconds

        return MockEvent()

    @classmethod
    def mock_cuda_function_stub(cls, *args, **kwargs):
        log_warning("Unsupported CUDA function called. Ignoring.", "torch.cuda", device_type=cls.get_default_device())

    @classmethod
    def mock_torch_stream(cls, device=None, priority=0):
        """Mock for torch.Stream class."""
        if device is None:
            device = cls.get_default_device()

        log_info("Creating torch.Stream for device %s with priority %s." % (device, priority), 
                "torch.Stream", device_type=device)

        # Return a simple object that mimics a Stream
        class MockStream:
            def __init__(self):
                self.device = device
                self.priority = priority

            def query(self):
                log_info("Stream query called.", "torch.Stream.query", device_type=device)
                return True

            def record_event(self, event=None):
                log_info("Stream record_event called.", "torch.Stream.record_event", device_type=device)
                if event is None:
                    # Create a mock event if none provided
                    return cls.mock_cuda_event()
                return event

            def synchronize(self):
                log_info("Stream synchronize called.", "torch.Stream.synchronize", device_type=device)

            def wait_event(self, event):
                log_info("Stream wait_event called.", "torch.Stream.wait_event", device_type=device)

            def wait_stream(self, stream):
                log_info("Stream wait_stream called.", "torch.Stream.wait_stream", device_type=device)

            def __eq__(self, other):
                return isinstance(other, MockStream) and self.device == other.device and self.priority == other.priority

        return MockStream()

    @classmethod
    def apply_patches(cls):
        """
        Apply patches to make CUDA functions work on MPS.
        """
        # Replace torch.device with our TorchDevice replacement
        torch.device = cls.create_device

        # Add support for torch.Stream
        if hasattr(torch, 'Stream'):
            torch.Stream = cls.mock_torch_stream

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
        torch.cuda.stream = cls.mock_cuda_stream
        torch.cuda.current_stream = cls.mock_cuda_current_stream
        torch.cuda.default_stream = cls.mock_cuda_default_stream
        torch.cuda.Event = cls.mock_cuda_event


        # Add Stream class to torch.cuda
        torch.cuda.Stream = lambda *args, **kwargs: cls.mock_cuda_stream()
        
        # Add stream context manager to torch.cuda
        def stream_context(stream=None):
            if stream is None:
                stream = cls.mock_cuda_stream()
            return stream
        torch.cuda.stream = stream_context
        # Override additional CUDA functions with mocks
        torch.cuda.reset_peak_memory_stats = cls.mock_cuda_reset_peak_memory_stats
        torch.cuda.ipc_collect = cls.mock_cuda_ipc_collect

        # Stub out unsupported CUDA functions
        unsupported_functions = [
            'CUDAGraph',
            'caching_allocator_alloc',
            'caching_allocator_delete',
            'can_device_access_peer',
            'change_current_allocator',
            'clock_rate',
            'comm',
            'current_blas_handle',
            'get_allocator_backend',
            'get_gencode_flags',
            'get_rng_state',
            'get_rng_state_all',
            'get_sync_debug_mode',
            'graph',    
            'graph_pool_handle',
            'initial_seed',
            'is_current_stream_capturing',
            'jiterator',
            'list_gpu_processes',
            'make_graphed_callables',
            'manual_seed',
            'manual_seed_all',
            'mem_get_info',
            'memory_usage',
            'nvtx',
            'power_draw',
            'reset_accumulated_memory_stats',
            'reset_max_memory_allocated',
            'reset_max_memory_cached',
            'seed',
            'seed_all',
            'set_rng_state',
            'set_rng_state_all',
            'set_stream',
            'set_sync_debug_mode',
            'temperature',
            'utilization',
        ]

        for func_name in unsupported_functions:
            if not hasattr(torch.cuda, func_name):
                continue
            setattr(torch.cuda, func_name, cls.mock_cuda_function_stub)

        # Patch torch.backends.cuda
        if not hasattr(torch, "backends"):
            torch.backends = types.ModuleType("backends")
        if not hasattr(torch.backends, "cuda"):
            torch.backends.cuda = types.ModuleType("cuda")

        torch.backends.cuda.is_built = lambda: True
        torch.backends.cuda.matmul = types.ModuleType("matmul")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        torch.backends.cuda.cufft = types.ModuleType("cufft")
        torch.backends.cuda.cufft.allow_tf32 = False

        torch.backends.cuda.enable_mem_efficient_sdp = lambda x: None
        torch.backends.cuda.enable_flash_sdp = lambda x: None
        torch.backends.cuda.enable_math_sdp = lambda x: None
        torch.backends.cuda.flash_attention_available = lambda: False
        torch.backends.cuda.sdp_kernel_available = lambda *args, **kwargs: False

# Apply patches when the module is imported
TorchDevice.apply_patches()