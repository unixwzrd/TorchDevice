"""
TorchDevice - Transparent PyTorch Device Redirection

This module enables seamless code portability between NVIDIA CUDA, Apple Silicon (MPS),
and CPU hardware for PyTorch applications. It intercepts PyTorch calls related to GPU
hardware, allowing developers to write code that works across different hardware
without modification.

Key features:
- Automatic device redirection based on available hardware
- CPU override capability using 'cpu:-1' device specification
- Mocked CUDA functions for MPS and CPU compatibility
- Stream and Event support across all device types
- Unified memory handling and reporting
- Detailed logging for debugging and migration assistance

Usage:
    import TorchDevice  # Import before torch to apply patches
    import torch
    
    # Regular device selection (will be redirected based on available hardware)
    device = torch.device('cuda')  # Redirects to MPS on Apple Silicon
    
    # Force CPU usage with the override feature
    device = torch.device('cpu:-1')  # Forces CPU regardless of available GPUs
    
    # All subsequent operations respect the CPU override
    tensor = torch.randn(5, 5)  # Will be created on CPU
    model = torch.nn.Linear(10, 5).to('cuda')  # Still uses CPU due to override
"""
import os
import threading
import time
import psutil
import torch
from .modules.TDLogger import auto_log, log_info  # We now use only auto_log instead of log_info for debugging

# Capture the original torch.device type.
_ORIGINAL_TORCH_DEVICE_TYPE = torch.device("cpu").__class__

_CACHED_DEFAULT_DEVICE = None
_device_type = None

# --- AMP Hooks ---
if hasattr(torch.cuda, 'amp'):
    _original_autocast = torch.cuda.amp.autocast

    @auto_log()
    def autocast_replacement(*args, **kwargs):
        default_device = TorchDevice.get_default_device()
        if default_device != 'cuda':
            return _original_autocast(*args, **kwargs)

    torch.cuda.amp.autocast = autocast_replacement

    if hasattr(torch.cuda.amp, 'GradScaler'):
        _OriginalGradScaler = torch.cuda.amp.GradScaler

        class GradScalerReplacement(_OriginalGradScaler):
            @auto_log()
            def __init__(self, *args, **kwargs):
                if TorchDevice.get_default_device() != 'cuda':
                    pass
                super().__init__(*args, **kwargs)
        torch.cuda.amp.GradScaler = GradScalerReplacement

# --- TorchDevice Class with Patched CUDA Functions ---
class TorchDevice:
    _default_device = None
    _previous_default_device = None
    _lock = threading.RLock()
    _cpu_override = False  # Flag for explicit CPU override

    _original_tensor_to = torch.Tensor.to
    _original_module_to = torch.nn.Module.to
    _original_module_cpu = torch.nn.Module.cpu
    _original_module_cuda = torch.nn.Module.cuda
    _original_tensor_cuda = torch.Tensor.cuda
    _original_torch_backends_cuda_is_built = torch.backends.cuda.is_built
    _original_torch_cuda_current_device = torch.cuda.current_device
    _original_torch_cuda_device = torch.cuda.device  # Context manager
    _original_torch_cuda_device_count = torch.cuda.device_count
    _original_torch_cuda_empty_cache = torch.cuda.empty_cache
    _original_torch_cuda_get_arch_list = torch.cuda.get_arch_list
    _original_torch_cuda_get_device_capability = torch.cuda.get_device_capability
    _original_torch_cuda_get_device_name = torch.cuda.get_device_name
    _original_torch_cuda_get_device_properties = torch.cuda.get_device_properties
    _original_torch_cuda_is_available = torch.cuda.is_available
    _original_torch_cuda_is_initialized = torch.cuda.is_initialized
    _original_torch_cuda_set_device = torch.cuda.set_device
    _original_torch_cuda_synchronize = torch.cuda.synchronize
    _original_torch_device = torch.device
    _original_torch_load = torch.load

    @auto_log()
    def __init__(self, device_type: str = None, device_index: int = None):
        with self._lock:
            if self._default_device is None:
                self.__class__._detect_default_device()
            if isinstance(device_type, str):
                if ':' in device_type:
                    device_type, index = device_type.split(':')
                    device_index = int(index)
                else:
                    device_index = 0 if device_index is None else device_index
                device_type = self.__class__._redirect_device_type(device_type)
                device_str = f"{device_type}:{device_index}"
                self.device = self.__class__._original_torch_device(device_str)

    @auto_log()
    def __repr__(self):
        return repr(self.device)

    @auto_log()
    def __str__(self):
        return str(self.device)

    class TorchDeviceWrapper(object):
        @auto_log()
        def __init__(self, device):
            self._device = device
            
        def __getattr__(self, name):
            return getattr(self._device, name)
            
        def __repr__(self):
            return repr(self._device)
            
        def __str__(self):
            return str(self._device)
            
        def __instancecheck__(self, instance):
            return isinstance(instance, _ORIGINAL_TORCH_DEVICE_TYPE)
    
    @classmethod
    @auto_log()
    def get_default_device(cls):
        """
        Return the default device based on available hardware and cache the result.
        """
        global _CACHED_DEFAULT_DEVICE
        if _CACHED_DEFAULT_DEVICE is None:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                _CACHED_DEFAULT_DEVICE = 'mps'
            elif cls._original_torch_cuda_is_available():
                _CACHED_DEFAULT_DEVICE = 'cuda'
            else:
                _CACHED_DEFAULT_DEVICE = 'cpu'
        return _CACHED_DEFAULT_DEVICE


    @classmethod
    def cpu_override_set(cls):
        return cls._cpu_override


    @classmethod
    @auto_log()
    def redirect_device_type(cls, device_type):
        """
        Redirect a device type string based on availability and CPU override.
        If cpu_override is True, always returns 'cpu'.
        For 'cuda' and 'mps' requests, return the type that is available.
        """
        # For explicit CPU requests, always return 'cpu'
        if device_type == 'cpu':
            return 'cpu'
        
        if device_type.startswith('cuda'):
            if _CACHED_DEFAULT_DEVICE == 'cuda':
                device_type = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_type = 'mps'
            else:
                device_type = 'cpu'
        elif device_type.startswith('mps'):
            if _CACHED_DEFAULT_DEVICE == 'mps':
                device_type = 'mps'
            elif cls._original_torch_cuda_is_available():
                device_type = 'cuda'
            else:
                device_type = 'cpu'
        return device_type

    @classmethod
    @auto_log()
    def tensor_creation_wrapper(cls, original_func):
        """
        Wrapper for tensor creation functions to enforce default device redirection and CPU override.
        Always calls torch_device_replacement for the device argument, so the toggle state is always respected.
        """
        def wrapped_func(*args, **kwargs):
            device_arg = kwargs.get('device', None)
            # If device is not specified, inject the current device (default or override)
            if device_arg is None:
                device = cls.torch_device_replacement()
                log_info(f"[tensor_creation_wrapper] Injecting device: {device}")
                kwargs['device'] = device
            else:
                # Always pass through torch_device_replacement to handle override logic
                device = cls.torch_device_replacement(device_arg)
                log_info(f"[tensor_creation_wrapper] Normalized device: {device}")
                kwargs['device'] = device
            return original_func(*args, **kwargs)
        return wrapped_func

    @staticmethod
    def tensor_to_replacement(t, *args, **kwargs):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"tensor_to_replacement called on non-tensor object: {type(t)}")
        if args and isinstance(args[0], (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
            device = TorchDevice.torch_device_replacement(args[0])
            new_args = (device,) + args[1:]
            kwargs.pop('device', None)
            return TorchDevice._original_tensor_to(t, *new_args, **kwargs)
        elif 'device' in kwargs and isinstance(kwargs['device'], (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
            device = TorchDevice.torch_device_replacement(kwargs['device'])
            kwargs['device'] = device
            return TorchDevice._original_tensor_to(t, *args, **kwargs)
        else:
            return TorchDevice._original_tensor_to(t, *args, **kwargs)

    @staticmethod
    def module_to_replacement(m, *args, **kwargs):
        if not isinstance(m, torch.nn.Module):
            raise TypeError(f"module_to_replacement called on non-module object: {type(m)}")
        if args and isinstance(args[0], (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
            device = TorchDevice.torch_device_replacement(args[0])
            new_args = (device,) + args[1:]
            kwargs.pop('device', None)
            return TorchDevice._original_module_to(m, *new_args, **kwargs)
        elif 'device' in kwargs and isinstance(kwargs['device'], (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
            device = TorchDevice.torch_device_replacement(kwargs['device'])
            kwargs['device'] = device
            return TorchDevice._original_module_to(m, *args, **kwargs)
        else:
            return TorchDevice._original_module_to(m, *args, **kwargs)

    @classmethod
    @auto_log()
    def torch_device_replacement(cls, *args, **kwargs) -> torch.device:
        """
        Drop-in replacement for torch.device() with device redirection and CPU override toggle.
        • No arguments → returns default device (or CPU if override is active).
        • 'cpu:-1' or torch.device('cpu', -1) → toggles CPU override.
        • Redirects non-CPU devices to available hardware.
        • Preserves extra args and kwargs.
        Always returns a torch.device object.
        """
        global _CACHED_DEFAULT_DEVICE
        log_info(f"Called with args={args}, kwargs={kwargs}")
        with cls._lock:
            if cls._default_device is None:
                cls._default_device = cls.get_default_device()
            # If no args, return device based on override
            if not args and not kwargs:
                if cls._cpu_override:
                    log_info("Override active, returning CPU device")
                    result = cls._original_torch_device('cpu')
                else:
                    default = cls.get_default_device()
                    log_info(f"No args, using default device: {default}")
                    result = cls._original_torch_device(default, 0) if default.lower() != "cpu" else cls._original_torch_device(default)
                assert isinstance(result, _ORIGINAL_TORCH_DEVICE_TYPE), f"Expected torch.device, got {type(result)}"
                return result

            # If first argument is torch.device, check for override
            if args and isinstance(args[0], _ORIGINAL_TORCH_DEVICE_TYPE):
                dev = args[0]
                if dev.type == 'cpu' and dev.index == -1:
                    if cls.cpu_override_set():
                        # Toggle OFF
                        cls._cpu_override = False
                        _CACHED_DEFAULT_DEVICE = cls._previous_default_device
                        cls._previous_device = None
                        log_info("CPU override toggled OFF (torch.device object)")
                    else:
                        # Toggle ON
                        cls._cpu_override = True
                        cls._previous_default_device = _CACHED_DEFAULT_DEVICE
                        _CACHED_DEFAULT_DEVICE = 'cpu'
                        log_info("CPU override toggled ON (torch.device object)")
                    return cls._original_torch_device('cpu')
                log_info(f"First arg is already torch.device: {args[0]}")
                assert isinstance(args[0], _ORIGINAL_TORCH_DEVICE_TYPE), f"Expected torch.device, got {type(args[0])}"
                return args[0]

            # If first argument is string device spec, parse and modify
            if args and isinstance(args[0], str):
                device_spec = args[0].strip()
                device_type = ""
                device_index = None

                if ":" in device_spec:
                    parts = device_spec.split(":", 1)
                    device_type = parts[0].lower()
                    try:
                        device_index = int(parts[1])
                    except ValueError:
                        device_index = None
                else:
                    device_type = device_spec.lower()

                # CPU override toggle logic
                if device_type == "cpu" and device_index == -1:
                    if cls.cpu_override_set():
                        # Toggle OFF
                        cls._cpu_override = False
                        _CACHED_DEFAULT_DEVICE = cls._previous_default_device
                        cls._previous_device = None
                        log_info("CPU override toggled OFF")
                        device_type = cls._default_device
                        device_index = 0 if device_type != "cpu" else None
                    else:
                        # Toggle ON
                        cls._cpu_override = True
                        cls._previous_default_device = _CACHED_DEFAULT_DEVICE
                        _CACHED_DEFAULT_DEVICE = 'cpu'
                        log_info("CPU override toggled ON")
                        device_type = "cpu"
                        device_index = 0

                # Always enforce CPU override for all device requests
                if cls.cpu_override_set():
                    device_type = "cpu"
                    device_index = None

                # Redirection logic (only applies if override is not active)
                elif device_type == "cpu":
                    device_type = cls.get_default_device()
                    device_index = 0 if device_type != "cpu" else None
                    log_info(f"Redirected 'cpu' to default device: {device_type}")
                elif device_type in ("cuda", "mps"):
                    redirected = cls.redirect_device_type(device_type)
                    log_info(f"Redirected device_type: {device_type} -> {redirected}")
                    device_type = redirected
                    device_index = 0 if device_type != "cpu" else None

                # Reassemble args
                new_arg = device_type
                if device_index is not None:
                    new_arg = f"{device_type}:{device_index}"
                args = (new_arg,) + args[1:]
                log_info(f"Reassembled args: {args}")

        # Always return a torch.device object
        result = cls._original_torch_device(*args, **kwargs)
        assert isinstance(result, _ORIGINAL_TORCH_DEVICE_TYPE), f"Expected torch.device, got {type(result)}"
        log_info(f"Returning device: {result}")
        return result

    @classmethod
    @auto_log()
    def torch_load_replacement(cls, *args, **kwargs):
        global _in_torch_load
        if _in_torch_load:
            return cls._original_torch_load(*args, **kwargs)
        _in_torch_load = True
        try:
            default_device = cls.get_default_device()
            if 'map_location' in kwargs:
                if kwargs['map_location'] == 'cpu' or (isinstance(kwargs['map_location'], str) and kwargs['map_location'] != default_device):
                    kwargs['map_location'] = default_device
            else:
                kwargs['map_location'] = default_device
            return cls._original_torch_load(*args, **kwargs)
        finally:
            _in_torch_load = False


    @classmethod
    @auto_log()
    def _detect_default_device(cls):
        if torch.backends.mps.is_available():
            cls._default_device = 'mps'
        elif cls._original_torch_cuda_is_available():
            cls._default_device = 'cuda'
        else:
            cls._default_device = 'cpu'


    @classmethod
    @auto_log()
    def tensor_cuda_replacement(cls, self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        default_device = cls.get_default_device()
        if default_device == 'mps':
            return self.to('mps', non_blocking=non_blocking, memory_format=memory_format)
        return cls._original_tensor_cuda(self, device, non_blocking, memory_format)

    @classmethod
    @auto_log()
    def module_cuda_replacement(cls, self, device=None):
        default_device = cls.get_default_device()
        if default_device == 'mps':
            return self.to('mps')
        return cls._original_module_cuda(self, device)


    @classmethod
    @auto_log()
    def mock_cuda_is_available(cls):
        return cls._default_device in ['cuda', 'mps']

    @classmethod
    @auto_log()
    def mock_cuda_device_count(cls):
        if cls._default_device == 'cuda':
            return cls._original_torch_cuda_device_count()
        elif cls._default_device == 'mps':
            return 1
        else:
            return 0

    @classmethod
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

    @classmethod
    @auto_log()
    def mock_cuda_memory_allocated(cls, device=None):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    @classmethod
    @auto_log()
    def mock_cuda_memory_reserved(cls, device=None):
        return psutil.virtual_memory().total

    @classmethod
    @auto_log()
    def mock_cuda_max_memory_allocated(cls, device=None):
        return cls.mock_cuda_memory_allocated(device)

    @classmethod
    @auto_log()
    def mock_cuda_max_memory_reserved(cls, device=None):
        return cls.mock_cuda_memory_reserved(device)

    @classmethod
    @auto_log()
    def mock_cuda_memory_stats(cls, device=None):
        stats = {
            'active.all.current': cls.mock_cuda_memory_allocated(device),
            'active.all.peak': cls.mock_cuda_max_memory_allocated(device),
            'reserved_bytes.all.current': cls.mock_cuda_memory_reserved(device),
            'reserved_bytes.all.peak': cls.mock_cuda_max_memory_reserved(device),
        }
        return stats

    @classmethod
    @auto_log()
    def mock_cuda_memory_snapshot(cls):
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
    @auto_log()
    def mock_cuda_memory_summary(cls, device=None, abbreviated=False):
        summary = f"Memory Allocated: {cls.mock_cuda_memory_allocated(device)} bytes\n"
        summary += f"Memory Reserved: {cls.mock_cuda_memory_reserved(device)} bytes\n"
        return summary

    @classmethod
    @auto_log()
    def mock_cuda_is_initialized(cls):
        return cls._default_device in ['cuda', 'mps']

    @classmethod
    @auto_log()
    def mock_cuda_get_arch_list(cls):
        if cls._default_device == 'cuda':
            return cls._original_torch_cuda_get_arch_list()
        elif cls._default_device == 'mps':
            return ['mps']
        else:
            return []

    @classmethod
    @auto_log()
    def mock_cuda_is_built(cls):
        if cls._default_device in ['cuda', 'mps']:
            return True
        else:
            return False

    @classmethod
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

    @classmethod
    @auto_log()
    def mock_cuda_empty_cache(cls):
        if cls._default_device == 'cuda':
            cls._original_torch_cuda_empty_cache()
        elif cls._default_device == 'mps':
            torch.mps.empty_cache()

    @classmethod
    @auto_log()
    def mock_cuda_synchronize(cls, device=None):
        if cls._default_device == 'cuda':
            cls._original_torch_cuda_synchronize(device)
        elif cls._default_device == 'mps':
            torch.mps.synchronize()

    @classmethod
    @auto_log()
    def mock_cuda_current_device(cls):
        if cls._default_device == 'cuda':
            return cls._original_torch_cuda_current_device()
        elif cls._default_device == 'mps':
            return 0
        else:
            return -1

    @classmethod
    @auto_log()
    def mock_cuda_set_device(cls, device):
        if cls._default_device == 'cuda':
            cls._original_torch_cuda_set_device(device)

    @classmethod
    @auto_log()
    def mock_cuda_get_device_name(cls, device=None):
        if cls._default_device == 'cuda':
            return cls._original_torch_cuda_get_device_name(device)
        elif cls._default_device == 'mps':
            return 'Apple MPS'
        else:
            return 'CPU'

    @classmethod
    @auto_log()
    def mock_cuda_get_device_capability(cls, device=None):
        if cls._default_device == 'cuda':
            return cls._original_torch_cuda_get_device_capability(device)
        elif cls._default_device == 'mps':
            return (0, 0)
        else:
            return (0, 0)

    @classmethod
    @auto_log()
    def mock_cuda_ipc_collect(cls):
        if cls._default_device == 'cuda':
            return torch.cuda.ipc_collect()

    @classmethod
    @auto_log()
    def mock_cuda_stream_class(cls, *args, **kwargs):
        try:
            from torch._streambase import _StreamBase
        except (AttributeError, ImportError):
            try:
                from torch._C import _StreamBase
            except (AttributeError, ImportError):
                try:
                    _StreamBase = torch._C._StreamBase
                except (AttributeError, ImportError):
                    _StreamBase = object
        
        class MPSStream(_StreamBase):
            @auto_log()
            def __init__(self, device=None, priority=0):
                if _StreamBase is not object:
                    try:
                        super().__init__()
                    except Exception:
                        pass
                self.device = device
                self.priority = priority
                self._is_created = True
                self._is_destroyed = False
            
            @auto_log()
            def synchronize(self):
                if cls._default_device == 'mps':
                    torch.mps.synchronize()
                return self
            
            @auto_log()
            def query(self):
                return True
            
            @auto_log()
            def wait_event(self, event):
                return self
            
            @auto_log()
            def wait_stream(self, stream):
                if hasattr(stream, 'synchronize'):
                    stream.synchronize()
                self.synchronize()
                return self
            
            @auto_log()
            def record_event(self, event=None):
                if event is None:
                    event = cls.mock_cuda_event(enable_timing=True)
                event.record(self)
                return event
            
            @auto_log()
            def __enter__(self):
                self._old_stream = torch.cuda.current_stream()
                return self
            
            @auto_log()
            def __exit__(self, exc_type, exc_val, exc_tb):
                return False
            
            @auto_log()
            def __str__(self):
                return f"MPSStream(device={self.device}, priority={self.priority})"
            
            @auto_log()
            def __eq__(self, o):
                if isinstance(o, MPSStream):
                    return (self.device == o.device and self.priority == o.priority)
                return False
                
            @auto_log()
            def __hash__(self):
                return hash((self.device, self.priority))
        
        device = kwargs.get('device', None)
        priority = kwargs.get('priority', 0)
        return MPSStream(device, priority)

    @classmethod
    @auto_log()
    def mock_cuda_event(cls, *args, **kwargs):
        enable_timing = kwargs.get('enable_timing', False)
        blocking = kwargs.get('blocking', False)
        interprocess = kwargs.get('interprocess', False)
        device = kwargs.get('device', None)
        MPSEvent = cls._get_mps_event_class()
        return MPSEvent(enable_timing=enable_timing, blocking=blocking, interprocess=interprocess, device=device)
    
    @classmethod
    @auto_log()
    def _get_mps_event_class(cls):
        try:
            from torch._streambase import _EventBase
        except (AttributeError, ImportError):
            try:
                from torch._C import _EventBase
            except (AttributeError, ImportError):
                try:
                    _EventBase = torch._C._EventBase
                except (AttributeError, ImportError):
                    _EventBase = object
        
        class MPSEvent(_EventBase):
            @auto_log()
            def __init__(self, enable_timing=False, blocking=False, interprocess=False, device=None):
                if _EventBase is not object:
                    try:
                        super().__init__()
                    except Exception:
                        pass
                self.enable_timing = enable_timing
                self.blocking = blocking
                self.interprocess = interprocess
                self.device = device
                self._is_created = True
                self._is_destroyed = False
                self._recorded = False
                self._record_time = None
                self._stream = None
            
            @auto_log()
            def record(self, stream=None):
                self._recorded = True
                self._record_time = time.time()
                self._stream = stream
                return self
            
            @auto_log()
            def wait(self, stream=None):
                return self
            
            @auto_log()
            def query(self):
                return self._recorded
            
            @auto_log()
            def elapsed_time(self, end_event):
                if not self.enable_timing:
                    return 0.5
                if not self._recorded or not getattr(end_event, '_recorded', False):
                    return 0.5
                start_time = self._record_time
                end_time = getattr(end_event, '_record_time', time.time())
                if start_time is None or end_time is None:
                    return 0.5
                elapsed_ms = (end_time - start_time) * 1000.0
                return elapsed_ms
            
            @auto_log()
            def synchronize(self):
                return self
            
            @auto_log()
            def __del__(self):
                if hasattr(self, '_is_destroyed') and not self._is_destroyed:
                    self._is_destroyed = True
            
        return MPSEvent
        
    @classmethod
    @auto_log()
    def mock_cuda_stream(cls, stream=None):
        class StreamContext:
            @auto_log()
            def __init__(self, stream):
                self.stream = stream
            
            @auto_log()
            def __enter__(self):
                if self.stream is not None and hasattr(self.stream, '__enter__'):
                    self.stream.__enter__()
                return self.stream
            
            @auto_log()
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.stream is not None and hasattr(self.stream, '__exit__'):
                    return self.stream.__exit__(exc_type, exc_val, exc_tb)
                return False
        
        return StreamContext(stream)

    @classmethod
    @auto_log()
    def mock_cuda_current_stream(cls, device=None):
        return cls.mock_cuda_stream_class(device=device)

    @classmethod
    @auto_log()
    def mock_cuda_default_stream(cls, device=None):
        return cls.mock_cuda_stream_class(device=device)

    @classmethod
    @auto_log()
    def mock_cuda_function_stub(cls, *args, **kwargs):
        pass

    @classmethod
    @auto_log()
    def mock_cuda_reset_peak_memory_stats(cls):
        pass

    @classmethod
    @auto_log()
    def apply_patches(cls):
        torch.device = cls.torch_device_replacement
        
        tensor_creation_functions = [
            'tensor', 'zeros', 'ones', 'empty', 'randn', 'rand', 'randint', 'arange', 'linspace', 'logspace'
        ]
        
        for func_name in tensor_creation_functions:
            if hasattr(torch, func_name):
                original_func = getattr(torch, func_name)
                setattr(torch, func_name, cls.tensor_creation_wrapper(original_func))

        MPSEvent = cls._get_mps_event_class()
        torch.Tensor.cuda = cls.tensor_cuda_replacement
        torch.backends.cuda.is_built = cls.mock_cuda_is_built
        torch.nn.Module.cuda = cls.module_cuda_replacement

        # Patch .to methods with standalone functions
        torch.Tensor.to = cls.tensor_to_replacement
        torch.nn.Module.to = cls.module_to_replacement

        # Patch .mps methods if present
        if hasattr(torch.Tensor, 'mps'):
            torch.Tensor.mps = cls.tensor_mps_replacement
        if hasattr(torch.nn.Module, 'mps'):
            torch.nn.Module.mps = cls.module_mps_replacement

        torch.cuda.Event = MPSEvent
        torch.cuda.Stream = cls.mock_cuda_stream_class
        torch.cuda.current_device = cls.mock_cuda_current_device
        torch.cuda.current_stream = cls.mock_cuda_current_stream
        torch.cuda.default_stream = cls.mock_cuda_default_stream
        torch.cuda.device = cls.mock_cuda_device_context
        torch.cuda.device_count = cls.mock_cuda_device_count
        torch.cuda.empty_cache = cls.mock_cuda_empty_cache
        torch.cuda.get_arch_list = cls.mock_cuda_get_arch_list
        torch.cuda.get_device_capability = cls.mock_cuda_get_device_capability
        torch.cuda.get_device_name = cls.mock_cuda_get_device_name
        torch.cuda.get_device_properties = cls.mock_cuda_get_device_properties
        torch.cuda.ipc_collect = cls.mock_cuda_ipc_collect
        torch.cuda.is_available = cls.mock_cuda_is_available
        torch.cuda.is_initialized = cls.mock_cuda_is_initialized
        torch.cuda.max_memory_allocated = cls.mock_cuda_max_memory_allocated
        torch.cuda.max_memory_reserved = cls.mock_cuda_max_memory_reserved
        torch.cuda.memory_allocated = cls.mock_cuda_memory_allocated
        torch.cuda.memory_reserved = cls.mock_cuda_memory_reserved
        torch.cuda.memory_snapshot = cls.mock_cuda_memory_snapshot
        torch.cuda.memory_stats = cls.mock_cuda_memory_stats
        torch.cuda.memory_summary = cls.mock_cuda_memory_summary
        torch.cuda.reset_peak_memory_stats = cls.mock_cuda_reset_peak_memory_stats
        torch.cuda.set_device = cls.mock_cuda_set_device
        torch.cuda.stream = cls.mock_cuda_stream
        torch.cuda.synchronize = cls.mock_cuda_synchronize

        torch.load = cls.torch_load_replacement

        torch.nn.Module.cuda = cls.module_cuda_replacement
        # .to methods already patched above
        
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
