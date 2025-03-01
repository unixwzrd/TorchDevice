import inspect
import logging
import os
import sys
import threading
import psutil  # For memory information
import torch
import time

# --- Capture Original torch.device Type ---
# This must be done before any patches to torch.device occur.
_ORIGINAL_TORCH_DEVICE_TYPE = torch.device("cpu").__class__

# --- Logging Setup and Helpers ---

# Set this to control verbosity: 0=minimal, 1=normal, 2=verbose
LOG_VERBOSITY = 1

logger = logging.getLogger('TorchDevice')
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(
    'GPU REDIRECT - [%(program_name)s] "%(caller_func_name)s" in File: %(caller_filename)s line: %(caller_lineno)s - Called: %(torch_function)s %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
# Prevent propagation to avoid duplicate messages
logger.propagate = False

# Track which messages have been logged to avoid duplicates
_logged_messages = set()
# Track the default device type
_device_type = "unknown"
# Track entry points to avoid internal chatter
_entry_points = {}

def get_caller_info():
    """
    Retrieve the caller's program name, module name, class name, line number, and function name.
    Examines the call stack to find the true entry point from external code into TorchDevice.
    """
    frame = inspect.currentframe()
    outer_frames = inspect.getouterframes(frame)
    program_name = os.path.basename(sys.argv[0])
    
    # Default return if we can't find a proper caller
    default_info = {
        'program_name': program_name,
        'module_name': 'UnknownModule',
        'caller_filename': 'TorchDevice.py',
        'class_name': 'N/A',
        'caller_lineno': 0,
        'caller_func_name': 'internal',
    }
    
    # We need at least one valid frame
    if len(outer_frames) <= 1:
        return default_info
    
    # Define patterns for internal frames to skip
    internal_patterns = [
        'TorchDevice.py', 
        'inspect.py', 
        'case.py', 
        'suite.py', 
        'runner.py',
        'torch/cuda',
        'torch/backends',
        'torch/_tensor.py',
        'torch/nn/modules/module.py'
    ]
    
    # Identify TorchDevice frames
    torchdevice_frames = []
    for i, frame_info in enumerate(outer_frames):
        if 'TorchDevice.py' in frame_info.filename:
            torchdevice_frames.append(i)
    
    # If no TorchDevice frames found, use the immediate caller
    if not torchdevice_frames:
        frame_info = outer_frames[1] if len(outer_frames) > 1 else outer_frames[0]
        return {
            'program_name': program_name,
            'module_name': inspect.getmodule(frame_info.frame).__name__ if inspect.getmodule(frame_info.frame) else 'UnknownModule',
            'caller_filename': os.path.basename(frame_info.filename),
            'class_name': 'N/A',
            'caller_lineno': frame_info.lineno,
            'caller_func_name': frame_info.function
        }
    
    # Find the highest (last) TorchDevice frame in the stack
    last_torchdevice_idx = max(torchdevice_frames) if torchdevice_frames else -1
    
    # If the last TorchDevice frame is the last frame in the stack, use default
    if last_torchdevice_idx >= len(outer_frames) - 1:
        return default_info
    
    # Start looking for user code after the last TorchDevice frame
    # If we couldn't find a TorchDevice frame, start from the second frame (skip this function)
    start_idx = last_torchdevice_idx + 1 if last_torchdevice_idx >= 0 else 1
    
    # Collect all frames after the last TorchDevice frame
    user_frames = []
    for i in range(start_idx, len(outer_frames)):
        frame_info = outer_frames[i]
        filename = frame_info.filename
        function = frame_info.function
        
        # Skip internal frames and framework infrastructure
        if any(pattern in filename for pattern in internal_patterns):
            continue
            
        # Skip unittest frames that aren't test methods
        if 'unittest' in filename and not function.startswith('test_'):
            continue
            
        # Found a user frame - collect it
        user_frames.append(i)
    
    # If we found any user frames, use the first one (closest to TorchDevice)
    if user_frames:
        frame_idx = user_frames[0]
        frame_info = outer_frames[frame_idx]
        
        # Extract information from the frame
        caller_filename = os.path.basename(frame_info.filename)
        lineno = frame_info.lineno
        func_name = frame_info.function
        
        # Get module name
        try:
            module = inspect.getmodule(frame_info.frame)
            module_name = module.__name__ if module else 'UnknownModule'
        except Exception:
            module_name = 'UnknownModule'
        
        # Get class name if available
        cls_name = 'N/A'
        try:
            if 'self' in frame_info.frame.f_locals:
                cls_name = frame_info.frame.f_locals['self'].__class__.__name__
            elif 'cls' in frame_info.frame.f_locals:
                cls_name = frame_info.frame.f_locals['cls'].__name__
        except Exception:
            pass
        
        return {
            'program_name': program_name,
            'module_name': module_name,
            'caller_filename': caller_filename,
            'class_name': cls_name,
            'caller_lineno': lineno,
            'caller_func_name': func_name
        }
    
    # If we didn't find a suitable frame, return the default
    return default_info

def log_message(level, message, torch_function=None):
    global _device_type, _entry_points
    
    # Skip logging based on verbosity level
    if LOG_VERBOSITY == 0 and level < logging.WARNING:
        return
    
    # Get caller information
    info = get_caller_info()
    
    # Skip internal TorchDevice calls completely unless they're warnings/errors
    if info['caller_filename'] == 'TorchDevice.py' and info['caller_func_name'] == 'internal':
        if level < logging.WARNING:
            return
    
    # Skip internal torch.device creation calls unless they're warnings/errors
    if torch_function == 'torch.device' and 'Creating torch.device' in message and level < logging.WARNING:
        # Only log device creation if it's explicitly requested by user code
        # Skip internal device creation for setup and initialization
        if 'setUp' in info['caller_func_name'] or 'init' in info['caller_func_name'].lower():
            return
    
    # Skip internal initialization messages unless they're warnings/errors
    if level < logging.WARNING:
        # Skip device detection messages
        if 'detected as default device' in message:
            return
        # Skip device setting messages
        if 'Default device set to:' in message:
            return
        # Skip tensor creation messages for internal operations
        if 'Creating tensor' in message and ('setUp' in info['caller_func_name'] or 'init' in info['caller_func_name'].lower()):
            return
        # Skip redirection messages for internal operations
        if 'Redirecting' in message and ('setUp' in info['caller_func_name'] or 'init' in info['caller_func_name'].lower()):
            return
    
    # Add torch_function to the info
    info['torch_function'] = torch_function if torch_function else 'unknown'
    
    # Add device type to the info
    info['device_type'] = _device_type
    
    # Create a unique key for this message to avoid duplicates
    caller_key = "%s:%s:%s" % (info['caller_filename'], info['caller_func_name'], info['caller_lineno'])
    function_key = torch_function if torch_function else 'unknown'
    message_key = "%s:%s:%s" % (caller_key, function_key, message)
    
    # Track the first time we see a torch function call from a specific location
    # Only track actual user code, not unittest internals
    if function_key not in _entry_points and 'unittest' not in info['caller_filename'] and 'case.py' not in info['caller_filename']:
        _entry_points[function_key] = caller_key
    
    # Determine if we should log this message
    is_first_call = _entry_points.get(function_key) == caller_key
    is_new_message = message_key not in _logged_messages
    is_warning_or_error = level >= logging.WARNING
    is_unittest = 'unittest' in info['caller_filename'] or 'case.py' in info['caller_filename']
    is_internal_call = ('TorchDevice.py' in info['caller_filename'] or 
                      'torch/cuda' in info['caller_filename'] or 
                      'torch/backends' in info['caller_filename'] or
                      'torch/_tensor.py' in info['caller_filename'] or
                      'torch/nn/modules/module.py' in info['caller_filename'])
    
    # Skip common internal operations that generate a lot of noise
    if not is_warning_or_error:
        # For redirection messages, only log the first occurrence from each location
        if 'Redirecting' in message and not is_first_call and LOG_VERBOSITY < 2:
            return False
            
        # Skip dunder method logs unless in verbose mode
        if torch_function and ('__' in torch_function or 'StreamContext' in torch_function) and LOG_VERBOSITY < 2:
            return False
            
        # Skip stream operation logs unless in verbose mode
        if 'stream' in torch_function.lower() and LOG_VERBOSITY < 2:
            return False
            
        # Skip event operation logs unless in verbose mode
        if 'event' in torch_function.lower() and LOG_VERBOSITY < 2:
            return False
    
    # For unittest frames, be more selective
    if is_unittest and not info['caller_func_name'].startswith('test_'):
        # Only log warnings/errors and first occurrences of messages from unittest internals
        should_log = is_warning_or_error or (is_new_message and LOG_VERBOSITY >= 2)
    elif is_internal_call and level < logging.WARNING:
        # For internal calls, only log warnings/errors unless in verbose mode
        should_log = LOG_VERBOSITY >= 2
    else:
        # For user code and test methods, log if:
        # 1. It's a warning or error
        # 2. It's the first time we've seen this function call from this location
        # 3. It's a new message we haven't seen before and verbosity > 0
        # 4. Verbosity is set to maximum (2)
        should_log = (
            is_warning_or_error or 
            (is_first_call and LOG_VERBOSITY > 0) or 
            (is_new_message and LOG_VERBOSITY > 0) or 
            LOG_VERBOSITY >= 2
        )
    
    if should_log:
        # Remember we've seen this message (unless in verbose mode)
        if LOG_VERBOSITY < 2:
            _logged_messages.add(message_key)
        
        # Format the message based on its content
        if message:
            # If it looks like it contains function arguments
            if any(x in message for x in ['(', '=', ',']) and not message.startswith('using:'):
                message = "using: %s" % message
        
        # Log the message
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
    global _CACHED_DEFAULT_DEVICE, _device_type
    if _CACHED_DEFAULT_DEVICE is None:
        if torch.backends.mps.is_available():
            _CACHED_DEFAULT_DEVICE = 'mps'
            _device_type = 'mps'
        elif torch.cuda.is_available():
            _CACHED_DEFAULT_DEVICE = 'cuda'
            _device_type = 'cuda'
        else:
            _CACHED_DEFAULT_DEVICE = 'cpu'
            _device_type = 'cpu'
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
        
        # Handle device redirection in kwargs
        if isinstance(device_arg, str):
            device_type = device_arg.split(':')[0] if ':' in device_arg else device_arg
            redirected_type = TorchDevice._redirect_device_type(device_type)
            
            if redirected_type != device_type:
                if ':' in device_arg:
                    index = device_arg.split(':')[1]
                    kwargs['device'] = f"{redirected_type}:{index}"
                else:
                    kwargs['device'] = redirected_type
                log_info(f"Redirecting tensor.to() from '{device_type}' to '{redirected_type}'.", "tensor.to")
        elif hasattr(device_arg, 'type'):
            device_type = device_arg.type
            redirected_type = TorchDevice._redirect_device_type(device_type)
            
            if redirected_type != device_type:
                index = getattr(device_arg, 'index', 0)
                if index is None:
                    index = 0
                kwargs['device'] = TorchDevice.torch_device_replacement(f"{redirected_type}:{index}")
                log_info(f"Redirecting tensor.to() from '{device_type}' to '{redirected_type}'.", "tensor.to")
    
    # Log warning if target device doesn't match default
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

    # Wrapper class for torch.device to handle isinstance checks properly
    class TorchDeviceWrapper(object):
        def __init__(self, device):
            self._device = device
            
        def __getattr__(self, name):
            return getattr(self._device, name)
            
        def __repr__(self):
            return repr(self._device)
            
        def __str__(self):
            return str(self._device)
            
        # This is needed for isinstance checks
        def __instancecheck__(self, instance):
            # Check if the instance is a torch.device
            return isinstance(instance, _ORIGINAL_TORCH_DEVICE_TYPE)
    
    @classmethod
    def torch_device_replacement(cls, device_type=None, device_index=None):
        with cls._lock:
            if cls._default_device is None:
                cls._detect_default_device()
            
            # Capture the original call details for better logging
            original_device_str = None
            if device_type is None:
                original_device_str = "None"
            elif isinstance(device_type, str):
                if device_index is not None:
                    original_device_str = f"{device_type}:{device_index}"
                else:
                    original_device_str = device_type
            else:
                original_device_str = str(device_type)
            
            # Process the device request
            if device_type is None:
                device_type = cls._default_device
                device_index = 0 if device_index is None else device_index
                device_str = f"{device_type}:{device_index}"
                log_info(f"Creating default device: torch.device('{device_str}')", "torch.device")
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
                    log_info(f"Redirecting torch.device('{original_device_str}') to torch.device('{device_str}')", "torch.device")
                else:
                    log_info(f"Creating torch.device('{device_str}')", "torch.device")
                
                return cls._original_torch_device(device_str)
            else:
                log_info(f"Creating torch.device with non-string type: {original_device_str}", "torch.device")
                return cls._original_torch_device(device_type)

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
        # Get the original device type for logging
        original_device_type = device_type
        
        if device_type.startswith('cuda'):
            if cls._default_device == 'cuda':
                return 'cuda'
            elif cls._default_device == 'mps':
                log_warning(f"CUDA device '{original_device_type}' requested but not available. Redirecting to MPS.", "torch.device")
                return 'mps'
            else:
                log_warning(f"CUDA device '{original_device_type}' requested but not available. Redirecting to CPU.", "torch.device")
                return 'cpu'
        elif device_type.startswith('mps'):
            if cls._default_device == 'mps':
                return 'mps'
            elif cls._default_device == 'cuda':
                log_warning(f"MPS device '{original_device_type}' requested but not available. Redirecting to CUDA.", "torch.device")
                return 'cuda'
            else:
                log_warning(f"MPS device '{original_device_type}' requested but not available. Redirecting to CPU.", "torch.device")
                return 'cpu'
        else:
            return device_type

    def __getattr__(self, attr):
        return getattr(self.device, attr)

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
    def mock_cuda_memory_stats(cls, device=None):
        log_info(f"torch.cuda.memory_stats called with device={device}", "torch.cuda.memory_stats")
        # Create a dictionary with the expected keys
        stats = {
            'active.all.current': cls.mock_cuda_memory_allocated(device),
            'active.all.peak': cls.mock_cuda_max_memory_allocated(device),
            'reserved_bytes.all.current': cls.mock_cuda_memory_reserved(device),
            'reserved_bytes.all.peak': cls.mock_cuda_max_memory_reserved(device),
        }
        return stats

    @classmethod
    def mock_cuda_memory_snapshot(cls):
        log_info("torch.cuda.memory_snapshot called", "torch.cuda.memory_snapshot")
        # Create a simple memory snapshot with minimal information
        snapshot = [
            {
                'device': 0,
                'address': 0,
                'total_size': cls.mock_cuda_memory_allocated(),
                'allocated_size': cls.mock_cuda_memory_allocated(),
                'active': True,
                'segment_type': 'small_pool',
            }
        ]
        return snapshot

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
    def mock_cuda_ipc_collect(cls):
        if cls._default_device == 'cuda':
            log_info("Collecting IPC memory.", "torch.cuda.ipc_collect")
            return torch.cuda.ipc_collect()
        else:
            log_warning("No GPU available to collect IPC memory.", "torch.cuda.ipc_collect")

    @classmethod
    def mock_cuda_stream_class(cls, *args, **kwargs):
        if LOG_VERBOSITY > 0:
            log_info(f"Creating CUDA stream with args={args}, kwargs={kwargs}", "torch.cuda.Stream")
        
        # Try to import _StreamBase from different possible locations
        try:
            from torch._streambase import _StreamBase
            log_info("Using torch._streambase._StreamBase as base class for MPSStream", "torch.cuda.Stream")
        except (AttributeError, ImportError):
            try:
                # Alternative way to get _StreamBase
                from torch._C import _StreamBase
                log_info("Using torch._C._StreamBase as base class for MPSStream", "torch.cuda.Stream")
            except (AttributeError, ImportError):
                try:
                    # Another alternative way
                    _StreamBase = torch._C._StreamBase
                    log_info("Using torch._C._StreamBase as base class for MPSStream (alternative method)", "torch.cuda.Stream")
                except (AttributeError, ImportError):
                    _StreamBase = object
                    log_warning("torch._streambase._StreamBase not found, using object as base class for MPSStream - this may cause issues with PyTorch dynamo", "torch.cuda.Stream")
        
        # Define the MPSStream class that inherits from _StreamBase
        class MPSStream(_StreamBase):
            def __init__(self, device=None, priority=0):
                # Call parent class constructor if it's not object
                if _StreamBase is not object:
                    try:
                        # Call the parent class constructor with the proper arguments
                        super().__init__()
                    except Exception as e:
                        log_warning(f"Error calling _StreamBase.__init__: {e}", "torch.cuda.Stream.__init__")
                
                self.device = device
                self.priority = priority
                self._is_created = True
                self._is_destroyed = False
                if LOG_VERBOSITY > 1:
                    log_info(f"MPSStream initialized with device={device}, priority={priority}", "torch.cuda.Stream.__init__")
            
            def synchronize(self):
                if LOG_VERBOSITY > 1:
                    log_info("MPSStream.synchronize called", "torch.cuda.Stream.synchronize")
                # Synchronize MPS device
                if cls._default_device == 'mps':
                    torch.mps.synchronize()
                return self
            
            def query(self):
                if LOG_VERBOSITY > 1:
                    log_info("MPSStream.query called", "torch.cuda.Stream.query")
                # Always return True for MPS streams as we can't query them
                return True
            
            def wait_event(self, event):
                if LOG_VERBOSITY > 1:
                    log_info(f"MPSStream.wait_event called with event={event}", "torch.cuda.Stream.wait_event")
                # In MPS, we don't need to call event.wait(self) as it causes an error
                # Just log the call and return self
                if not getattr(event, '_recorded', True):
                    log_warning("Event has not been recorded yet", "torch.cuda.Stream.wait_event")
                return self
            
            def wait_stream(self, stream):
                if LOG_VERBOSITY > 1:
                    log_info(f"MPSStream.wait_stream called with stream={stream}", "torch.cuda.Stream.wait_stream")
                # For MPS, just synchronize both streams
                if hasattr(stream, 'synchronize'):
                    stream.synchronize()
                self.synchronize()
                return self
            
            def record_event(self, event=None):
                if LOG_VERBOSITY > 1:
                    log_info(f"MPSStream.record_event called with event={event}", "torch.cuda.Stream.record_event")
                if event is None:
                    event = cls.mock_cuda_event(enable_timing=True)
                event.record(self)
                return event
            
            def __enter__(self):
                if LOG_VERBOSITY > 1:
                    log_info("MPSStream.__enter__ called", "torch.cuda.Stream.__enter__")
                # Store the current stream to restore it later
                self._old_stream = torch.cuda.current_stream()
                # Set this stream as current
                # Note: MPS doesn't support this, but we'll log it
                if LOG_VERBOSITY > 1:
                    log_info(f"Setting stream {self} as current", "torch.cuda.Stream.__enter__")
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if LOG_VERBOSITY > 1:
                    log_info("MPSStream.__exit__ called", "torch.cuda.Stream.__exit__")
                    # Restore the previous stream
                    log_info(f"Restoring stream {self._old_stream}", "torch.cuda.Stream.__exit__")
                return False  # Don't suppress exceptions
            
            def __del__(self):
                if hasattr(self, '_is_destroyed') and not self._is_destroyed:
                    self._is_destroyed = True
                    if LOG_VERBOSITY > 1:
                        log_info("MPSStream destroyed", "torch.cuda.Stream.__del__")
            
            def __str__(self):
                return f"MPSStream(device={self.device}, priority={self.priority})"
            
            def __eq__(self, o):
                if isinstance(o, MPSStream):
                    return (self.device == o.device and 
                            self.priority == o.priority)
                return False
                
            def __hash__(self):
                return hash((self.device, self.priority))
        
        device = kwargs.get('device', None)
        priority = kwargs.get('priority', 0)
        return MPSStream(device, priority)

    @classmethod
    def mock_cuda_event(cls, *args, **kwargs):
        enable_timing = kwargs.get('enable_timing', False)
        blocking = kwargs.get('blocking', False)
        interprocess = kwargs.get('interprocess', False)
        device = kwargs.get('device', None)
        
        if LOG_VERBOSITY > 0:
            log_info(f"Creating CUDA event with enable_timing={enable_timing}, blocking={blocking}", "torch.cuda.Event")
        
        # Create an instance of the MPSEvent class
        MPSEvent = cls._get_mps_event_class()
        return MPSEvent(enable_timing=enable_timing, blocking=blocking, interprocess=interprocess, device=device)
    
    @classmethod
    def _get_mps_event_class(cls):
        """Return the MPSEvent class that inherits from _EventBase.
        This is critical for compatibility with PyTorch's dynamo module.
        """
        # Get the _EventBase class from torch._streambase
        try:
            from torch._streambase import _EventBase
            log_info("Using torch._streambase._EventBase as base class for MPSEvent", "torch.cuda.Event")
        except (AttributeError, ImportError):
            try:
                # Alternative way to get _EventBase
                from torch._C import _EventBase
                log_info("Using torch._C._EventBase as base class for MPSEvent", "torch.cuda.Event")
            except (AttributeError, ImportError):
                try:
                    # Another alternative way
                    _EventBase = torch._C._EventBase
                    log_info("Using torch._C._EventBase as base class for MPSEvent (alternative method)", "torch.cuda.Event")
                except (AttributeError, ImportError):
                    _EventBase = object
                    log_warning("torch._streambase._EventBase not found, using object as base class for MPSEvent - this may cause issues with PyTorch dynamo", "torch.cuda.Event")
        
        # Define the MPSEvent class that inherits from _EventBase
        class MPSEvent(_EventBase):
            def __init__(self, enable_timing=False, blocking=False, interprocess=False, device=None):
                # Call parent class constructor if it's not object
                if _EventBase is not object:
                    try:
                        # Call the parent class constructor with the proper arguments
                        super().__init__()
                    except Exception as e:
                        log_warning(f"Error calling _EventBase.__init__: {e}", "torch.cuda.Event.__init__")
                
                self.enable_timing = enable_timing
                self.blocking = blocking
                self.interprocess = interprocess
                self.device = device
                self._is_created = True
                self._is_destroyed = False
                self._recorded = False
                self._record_time = None
                self._stream = None
                if LOG_VERBOSITY > 1:
                    log_info("MPSEvent initialized", "torch.cuda.Event.__init__")
            
            def record(self, stream=None):
                if LOG_VERBOSITY > 1:
                    log_info(f"MPSEvent.record called with stream={stream}", "torch.cuda.Event.record")
                self._recorded = True
                self._record_time = time.time()
                self._stream = stream
                return self
            
            def wait(self, stream=None):
                if LOG_VERBOSITY > 1:
                    log_info(f"MPSEvent.wait called with stream={stream}", "torch.cuda.Event.wait")
                if not self._recorded:
                    log_warning("Event has not been recorded yet", "torch.cuda.Event.wait")
                return self
            
            def query(self):
                if LOG_VERBOSITY > 1:
                    log_info("MPSEvent.query called", "torch.cuda.Event.query")
                return self._recorded
            
            def elapsed_time(self, end_event):
                if LOG_VERBOSITY > 1:
                    log_info(f"MPSEvent.elapsed_time called with end_event={end_event}", "torch.cuda.Event.elapsed_time")
                
                # Check if timing is enabled
                if not self.enable_timing:
                    log_warning("Events were created without timing enabled, but returning mock value anyway", "torch.cuda.Event.elapsed_time")
                    return 0.5  # Return a mock value even if timing is not enabled to avoid errors
                
                # Check if events have been recorded
                if not self._recorded or not getattr(end_event, '_recorded', False):
                    log_warning("One or both events have not been recorded, returning mock value", "torch.cuda.Event.elapsed_time")
                    return 0.5  # Return a mock value even if events are not recorded to avoid errors
                
                # Calculate elapsed time in milliseconds
                start_time = self._record_time
                end_time = getattr(end_event, '_record_time', time.time())
                
                # Safety check for None values
                if start_time is None or end_time is None:
                    log_warning("Event timestamps are None, returning mock value", "torch.cuda.Event.elapsed_time")
                    return 0.5
                
                elapsed_ms = (end_time - start_time) * 1000.0
                if LOG_VERBOSITY > 1:
                    log_info(f"Elapsed time: {elapsed_ms} ms", "torch.cuda.Event.elapsed_time")
                return elapsed_ms
            
            def synchronize(self):
                if LOG_VERBOSITY > 1:
                    log_info("MPSEvent.synchronize called", "torch.cuda.Event.synchronize")
                if not self._recorded:
                    log_warning("Event has not been recorded yet", "torch.cuda.Event.synchronize")
                return self
            
            def __del__(self):
                if hasattr(self, '_is_destroyed') and not self._is_destroyed:
                    self._is_destroyed = True
                    if LOG_VERBOSITY > 1:
                        log_info("MPSEvent destroyed", "torch.cuda.Event.__del__")
        
        return MPSEvent
        
        # The code below is no longer needed as we're using the MPSEvent class from _get_mps_event_class
        # Keeping this comment for reference
        
        return MPSEvent(enable_timing=enable_timing, blocking=blocking, interprocess=interprocess, device=device)

    @classmethod
    def mock_cuda_stream(cls, stream=None):
        if LOG_VERBOSITY > 0:
            log_info(f"torch.cuda.stream called with stream={stream}", "torch.cuda.stream")
        
        class StreamContext:
            def __init__(self, stream):
                self.stream = stream
                if LOG_VERBOSITY > 1:
                    log_info(f"StreamContext initialized with stream={stream}", "torch.cuda.stream.__init__")
            
            def __enter__(self):
                if LOG_VERBOSITY > 1:
                    log_info("StreamContext.__enter__ called", "torch.cuda.stream.__enter__")
                if self.stream is not None and hasattr(self.stream, '__enter__'):
                    self.stream.__enter__()
                return self.stream
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if LOG_VERBOSITY > 1:
                    log_info("StreamContext.__exit__ called", "torch.cuda.stream.__exit__")
                if self.stream is not None and hasattr(self.stream, '__exit__'):
                    return self.stream.__exit__(exc_type, exc_val, exc_tb)
                return False
        
        return StreamContext(stream)

    @classmethod
    def mock_cuda_current_stream(cls, device=None):
        if LOG_VERBOSITY > 0:
            log_info(f"torch.cuda.current_stream called with device={device}", "torch.cuda.current_stream")
        # Return a default stream for the device
        return cls.mock_cuda_stream_class(device=device)

    @classmethod
    def mock_cuda_default_stream(cls, device=None):
        log_info(f"torch.cuda.default_stream called with device={device}", "torch.cuda.default_stream")
        # Return a default stream for the device
        return cls.mock_cuda_stream_class(device=device)

    @classmethod
    def mock_cuda_function_stub(cls, *args, **kwargs):
        log_warning("Unsupported CUDA function called. Ignoring.", "torch.cuda")

    @classmethod
    def mock_cuda_reset_peak_memory_stats(cls):
        if cls._default_device in ['cuda', 'mps']:
            log_info("Resetting peak memory stats.", "torch.cuda.reset_peak_memory_stats")
        else:
            log_warning("No GPU available to reset peak memory stats.", "torch.cuda.reset_peak_memory_stats")

    @classmethod
    def tensor_creation_wrapper(cls, original_func):
        """Wrapper for tensor creation functions to handle device redirection"""
        def wrapped_func(*args, **kwargs):
            # Check if device is specified
            if 'device' in kwargs:
                device_arg = kwargs['device']
                # Handle different device argument types
                if isinstance(device_arg, str):
                    # For string device arguments
                    device_type = device_arg.split(':')[0] if ':' in device_arg else device_arg
                    
                    # Redirect the device if needed
                    redirected_type = cls._redirect_device_type(device_type)
                    
                    # If redirection happened, update the device argument
                    if redirected_type != device_type:
                        # For strings, replace the device type
                        if ':' in device_arg:
                            index = device_arg.split(':')[1]
                            kwargs['device'] = f"{redirected_type}:{index}"
                        else:
                            kwargs['device'] = redirected_type
                        
                        # Only log this message if we're in verbose mode
                        if LOG_VERBOSITY > 0:
                            log_info(f"Redirecting tensor creation from '{device_type}' to '{redirected_type}'.", original_func.__name__)
                elif hasattr(device_arg, 'type'):
                    # For device objects
                    device_type = device_arg.type
                    
                    # Redirect the device if needed
                    redirected_type = cls._redirect_device_type(device_type)
                    
                    # If redirection happened, update the device argument
                    if redirected_type != device_type:
                        # Create a new device with the redirected type
                        index = getattr(device_arg, 'index', 0)
                        if index is None:
                            index = 0
                        kwargs['device'] = cls.torch_device_replacement(f"{redirected_type}:{index}")
                        
                        # Only log this message if we're in verbose mode
                        if LOG_VERBOSITY > 0:
                            log_info(f"Redirecting tensor creation from '{device_type}' to '{redirected_type}'.", original_func.__name__)
            
            # Call the original function with potentially modified kwargs
            return original_func(*args, **kwargs)
        
        return wrapped_func

    @classmethod
    def apply_patches(cls):
        # Create a wrapper for torch.device class to handle isinstance checks
        original_isinstance = isinstance
        def patched_isinstance(obj, class_or_tuple):
            # Handle the case where torch.device is in the class_or_tuple
            if class_or_tuple == torch.device:
                # Check if obj is a torch.device instance
                return original_isinstance(obj, _ORIGINAL_TORCH_DEVICE_TYPE)
            elif original_isinstance(class_or_tuple, tuple):
                # Check if torch.device is in the tuple
                if torch.device in class_or_tuple:
                    return original_isinstance(obj, _ORIGINAL_TORCH_DEVICE_TYPE)
            # Otherwise, use the original isinstance
            return original_isinstance(obj, class_or_tuple)
        
        # Replace the built-in isinstance with our patched version
        builtins = __import__('builtins')
        builtins.isinstance = patched_isinstance
        
        # Replace torch.device with our TorchDevice replacement
        torch.device = cls.torch_device_replacement
        
        # Patch tensor creation functions
        tensor_creation_functions = [
            'tensor', 'zeros', 'ones', 'empty', 'randn', 'rand', 'randint', 'arange', 'linspace', 'logspace'
        ]
        
        for func_name in tensor_creation_functions:
            if hasattr(torch, func_name):
                original_func = getattr(torch, func_name)
                setattr(torch, func_name, cls.tensor_creation_wrapper(original_func))
        
        # Override CUDA functions with mocks
        torch.cuda.is_available = cls.mock_cuda_is_available
        torch.cuda.device_count = cls.mock_cuda_device_count
        torch.cuda.get_device_properties = cls.mock_cuda_get_device_properties
        torch.cuda.empty_cache = cls.mock_cuda_empty_cache
        torch.cuda.synchronize = cls.mock_cuda_synchronize
        torch.cuda.current_device = cls.mock_cuda_current_device
        torch.cuda.set_device = cls.mock_cuda_set_device
        torch.cuda.get_device_name = cls.mock_cuda_get_device_name
        torch.cuda.get_device_capability = cls.mock_cuda_get_device_capability
        torch.cuda.memory_allocated = cls.mock_cuda_memory_allocated
        torch.cuda.memory_reserved = cls.mock_cuda_memory_reserved
        torch.cuda.max_memory_allocated = cls.mock_cuda_max_memory_allocated
        torch.cuda.max_memory_reserved = cls.mock_cuda_max_memory_reserved
        torch.cuda.memory_stats = cls.mock_cuda_memory_stats
        torch.cuda.memory_snapshot = cls.mock_cuda_memory_snapshot
        torch.cuda.memory_summary = cls.mock_cuda_memory_summary
        torch.cuda.is_initialized = cls.mock_cuda_is_initialized
        torch.cuda.get_arch_list = cls.mock_cuda_get_arch_list
        torch.backends.cuda.is_built = cls.mock_cuda_is_built
        torch.cuda.device = cls.mock_cuda_device_context  # Override the context manager

        # Stream and Event related functions
        torch.cuda.Stream = cls.mock_cuda_stream_class
        torch.cuda.stream = cls.mock_cuda_stream
        torch.cuda.current_stream = cls.mock_cuda_current_stream
        torch.cuda.default_stream = cls.mock_cuda_default_stream
        
        # Get the MPSEvent class from our _get_mps_event_class method
        # This ensures we have a class that inherits from _EventBase for PyTorch dynamo compatibility
        MPSEvent = cls._get_mps_event_class()
        
        # Assign the MPSEvent class (not an instance) to torch.cuda.Event
        # This is critical for compatibility with PyTorch's dynamo module
        torch.cuda.Event = MPSEvent

        # Override additional CUDA functions with mocks
        torch.cuda.reset_peak_memory_stats = cls.mock_cuda_reset_peak_memory_stats
        torch.cuda.ipc_collect = cls.mock_cuda_ipc_collect
        
        # Stub out unsupported CUDA functions
        unsupported_functions = [
            'set_stream',
            'mem_get_info',
            'reset_accumulated_memory_stats',
            'reset_max_memory_allocated',
            'reset_max_memory_cached',
            'caching_allocator_alloc',
            'caching_allocator_delete',
            'get_allocator_backend',
            'change_current_allocator',
            'nvtx',
            'jiterator',
            'graph',
            'CUDAGraph',
            'make_graphed_callables',
            'is_current_stream_capturing',
            'graph_pool_handle',
            'can_device_access_peer',
            'comm',
            'get_gencode_flags',
            'current_blas_handle',
            'memory_usage',
            'utilization',
            'temperature',
            'power_draw',
            'clock_rate',
            'set_sync_debug_mode',
            'get_sync_debug_mode',
            'list_gpu_processes',
            'seed',
            'seed_all',
            'manual_seed',
            'manual_seed_all',
            'get_rng_state',
            'get_rng_state_all',
            'set_rng_state',
            'set_rng_state_all',
            'initial_seed',
        ]
        for func_name in unsupported_functions:
            if hasattr(torch.cuda, func_name):
                setattr(torch.cuda, func_name, cls.mock_cuda_function_stub)

# Apply patches when the module is imported
TorchDevice.apply_patches()