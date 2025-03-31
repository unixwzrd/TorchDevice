"""
Moscked CUDA functions
"""
import os
import time
import psutil
import torch
from .TDLogger import log_info, auto_log
from .device_detection import _CACHED_DEFAULT_DEVICE, _ORIGINAL_TORCH_DEVICE_TYPE

# Add diagnostic logging to understand type checking
def _debug_type_info(obj):
    """Helper to print detailed type information"""
    log_info(f"Object: {obj}")
    log_info(f"Object type: {type(obj)}")
    log_info(f"Object __class__: {obj.__class__}")
    if hasattr(obj, '__class__.__mro__'):
        log_info(f"MRO: {obj.__class__.__mro__}")
    return obj

# Save original device type
_original_torch_cuda_device = torch.cuda.device

# Log what torch.cuda.device is before we modify it
log_info(f"Original torch.cuda.device: {_original_torch_cuda_device}")
log_info(f"Original torch.cuda.device type: {type(_original_torch_cuda_device)}")

# Mock device class that matches PyTorch's device behavior
class _MockDevice:
    def __init__(self, index=None):
        log_info(f"Creating _MockDevice with index={index}")
        self.idx = index
        self._type = 'cuda'
    
    @property
    def type(self):
        return self._type
        
    @property
    def index(self):
        return self.idx

    def __str__(self):
        return f"{self.type}:{self.index}"

    def __repr__(self):
        return f"device(type='{self.type}', index={self.index})"

    def __eq__(self, other):
        if isinstance(other, _ORIGINAL_TORCH_DEVICE_TYPE):
            return (self.type == other.type and self.index == other.index)
        return False

    def __hash__(self):
        return hash((self.type, self.index))

    def __instancecheck__(self, instance):
        log_info(f"_MockDevice.__instancecheck__ called with {instance}")
        log_info(f"Checking against _ORIGINAL_TORCH_DEVICE_TYPE: {_ORIGINAL_TORCH_DEVICE_TYPE}")
        # If the instance is already a torch.device, it's valid
        if isinstance(instance, _ORIGINAL_TORCH_DEVICE_TYPE):
            log_info("Instance is already a torch.device")
            return True
        # If it's our mock device, check if it matches the device type
        if isinstance(instance, _MockDevice):
            log_info("Instance is a _MockDevice")
            return True
        log_info(f"Instance check failed for {instance}")
        return False

# Only add the mock if torch.cuda.device isn't already defined
if not hasattr(torch.cuda, 'device'):
    log_info("Setting up mock torch.cuda.device")
    # Create a type object for proper type checking
    device_type = type('device', (), {
        '__module__': 'torch.cuda',
        '__instancecheck__': lambda self, instance: (
            isinstance(instance, _ORIGINAL_TORCH_DEVICE_TYPE) or 
            isinstance(instance, _MockDevice)
        )
    })
    torch.cuda.device = device_type

# Log what we've set up
log_info(f"Final torch.cuda.device: {torch.cuda.device}")
_debug_type_info(torch.cuda.device)
test_device = _MockDevice(0)
log_info(f"Test device: {test_device}")
_debug_type_info(test_device)
log_info(f"Is test device instance of torch.cuda.device? {isinstance(test_device, torch.cuda.device)}")

@auto_log()
def mock_cuda_is_available(default_device):
    return default_device in ['cuda', 'mps']

@auto_log()
def mock_cuda_device_count(default_device):
    if default_device == 'cuda':
        return torch.cuda.device_count()
    elif default_device == 'mps':
        return 1
    else:
        return 0

@auto_log()
def mock_cuda_get_device_properties(default_device, device):
    if default_device == 'cuda':
        return torch.cuda.get_device_properties(device)
    elif default_device in ['mps', 'cpu']:
        class DummyDeviceProperties:
            name = 'Dummy GPU'
            total_memory = psutil.virtual_memory().total
            major = 0
            minor = 0
            multi_processor_count = 1
            def __str__(self):
                return f"DummyDeviceProperties(name={self.name}, total_memory={self.total_memory})"
        return DummyDeviceProperties()
    else:
        raise RuntimeError(f"Invalid default device: {default_device}")

@auto_log()
def mock_cuda_memory_allocated(default_device, device=None):
    """Get current memory allocated."""
    process = psutil.Process(os.getpid())
    if default_device == 'mps':
        # For MPS, we track RSS (Resident Set Size) as it represents actual physical memory used
        # This includes both CPU and GPU memory due to unified memory architecture
        return process.memory_info().rss
    else:
        return process.memory_info().rss

@auto_log()
def mock_cuda_memory_reserved(default_device, device=None):
    """Get current memory reserved."""
    if default_device == 'mps':
        # For MPS, we use the total memory as the reserved memory
        # since it's shared between CPU and GPU
        return psutil.virtual_memory().total
    return psutil.virtual_memory().total

@auto_log()
def mock_cuda_max_memory_allocated(default_device, device=None):
    """Get peak memory allocated."""
    if default_device == 'mps':
        # For MPS, we track the peak RSS of the process
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    return mock_cuda_memory_allocated(default_device, device)

@auto_log()
def mock_cuda_max_memory_reserved(default_device, device=None):
    """Get peak memory reserved."""
    if default_device == 'mps':
        # For MPS, total system memory is the maximum that could be reserved
        return psutil.virtual_memory().total
    return mock_cuda_memory_reserved(default_device, device)

@auto_log()
def mock_cuda_memory_stats(default_device, device=None):
    """Get comprehensive memory statistics."""
    process = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    
    if default_device == 'mps':
        # For MPS, provide more detailed memory stats
        return {
            'active.all.current': process.memory_info().rss,
            'active.all.peak': process.memory_info().rss,
            'reserved_bytes.all.current': vm.total,
            'reserved_bytes.all.peak': vm.total,
            'system.used': vm.used,
            'system.free': vm.free,
            'process.physical': process.memory_info().rss,
            'process.virtual': process.memory_info().vms,
        }
    return {
        'active.all.current': mock_cuda_memory_allocated(default_device, device),
        'active.all.peak': mock_cuda_max_memory_allocated(default_device, device),
        'reserved_bytes.all.current': mock_cuda_memory_reserved(default_device, device),
        'reserved_bytes.all.peak': mock_cuda_max_memory_reserved(default_device, device),
    }

@auto_log()
def mock_cuda_memory_snapshot(default_device):
    return [{
        'device': 0,
        'address': 0,
        'total_size': mock_cuda_memory_allocated(default_device),
        'allocated_size': mock_cuda_memory_allocated(default_device),
        'active': True,
        'segment_type': 'small_pool',
    }]

@auto_log()
def mock_cuda_memory_summary(default_device, device=None, abbreviated=False):
    return (f"Memory Allocated: {mock_cuda_memory_allocated(default_device, device)} bytes\n"
            f"Memory Reserved: {mock_cuda_memory_reserved(default_device, device)} bytes\n")

@auto_log()
def mock_cuda_is_initialized(default_device):
    return default_device in ['cuda', 'mps']

@auto_log()
def mock_cuda_get_arch_list(default_device):
    if default_device == 'cuda':
        return torch.cuda.get_arch_list()
    elif default_device == 'mps':
        return ['mps']
    else:
        return []

@auto_log()
def mock_cuda_is_built(default_device):
    return default_device in ['cuda', 'mps']

@auto_log()
def mock_cuda_device_context(default_device, device=None):
    class DeviceContextManager:
        @auto_log()
        def __init__(self, device):
            self.device = device
        @auto_log()
        def __enter__(self):
            mock_cuda_set_device(default_device, self.device)
        @auto_log()
        def __exit__(self, exc_type, exc_value, traceback):
            pass
    return DeviceContextManager(device)

@auto_log()
def mock_cuda_empty_cache(default_device):
    if default_device == 'cuda':
        torch.cuda.empty_cache()
    elif default_device == 'mps':
        torch.mps.empty_cache()
    else:
        pass

@auto_log()
def mock_cuda_synchronize(default_device, device=None):
    if default_device == 'cuda':
        torch.cuda.synchronize(device)
    elif default_device == 'mps':
        torch.mps.synchronize()
    else:
        pass

@auto_log()
def mock_cuda_current_device(default_device):
    if default_device == 'cuda':
        return torch.cuda.current_device()
    elif default_device == 'mps':
        return 0
    else:
        return -1

@auto_log()
def mock_cuda_set_device(default_device, device):
    if default_device == 'cuda':
        torch.cuda.set_device(device)
    elif default_device == 'mps':
        pass
    else:
        pass

@auto_log()
def mock_cuda_get_device_name(default_device, device=None):
    if default_device == 'cuda':
        return torch.cuda.get_device_name(device)
    elif default_device == 'mps':
        return 'Apple MPS'
    else:
        return 'CPU'

@auto_log()
def mock_cuda_get_device_capability(default_device, device=None):
    if default_device == 'cuda':
        return torch.cuda.get_device_capability(device)
    elif default_device == 'mps':
        return (0, 0)
    else:
        return (0, 0)

@auto_log()
def mock_cuda_ipc_collect(default_device):
    if default_device == 'cuda':
        return torch.cuda.ipc_collect()
    else:
        pass

@auto_log()
def mock_cuda_stream_class(default_device, *args, **kwargs):
    """Create a CUDA stream."""
    if default_device == 'cuda':
        return torch.cuda.Stream(*args, **kwargs)

    # Base stream class for MPS
    class MPSStream:
        def __init__(self, device=None, priority=0):
            self.device = device
            self.priority = priority
            self._is_created = True
            self._is_destroyed = False

        def synchronize(self):
            if default_device == 'mps':
                import torch.mps
                torch.mps.synchronize()
            return self

        def query(self):
            # MPS operations are implicitly synchronized
            return True

        def wait_event(self, event):
            if not getattr(event, '_recorded', True):
                return self
            # For MPS, waiting for an event means synchronizing the device
            if default_device == 'mps':
                import torch.mps
                torch.mps.synchronize()
            return self

        def wait_stream(self, stream):
            # For MPS, waiting for a stream means synchronizing both streams
            if hasattr(stream, 'synchronize'):
                stream.synchronize()
            self.synchronize()
            return self

        def record_event(self, event=None):
            if event is None:
                event = mock_cuda_event(default_device, enable_timing=True)
            event.record(self)
            return event

        def __enter__(self):
            self._old_stream = torch.cuda.current_stream()
            return self

        def __exit__(self, exc_type, exc_val, traceback):
            return False

        def __del__(self):
            if hasattr(self, '_is_destroyed') and not self._is_destroyed:
                self._is_destroyed = True

        def __str__(self):
            return f"MPSStream(device={self.device}, priority={self.priority})"

        def __eq__(self, o):
            if isinstance(o, MPSStream):
                return (self.device == o.device and self.priority == o.priority)
            return False

        def __hash__(self):
            return hash((self.device, self.priority))

    device_arg = kwargs.get('device', None)
    priority = kwargs.get('priority', 0)
    return MPSStream(device_arg, priority)

@auto_log()
def mock_cuda_reset_peak_memory_stats(default_device):
    """Reset peak memory stats."""
    # Add diagnostic logging
    log_info(f"torch.cuda.device is: {torch.cuda.device}")
    log_info(f"type of torch.cuda.device is: {type(torch.cuda.device)}")
    log_info(f"_ORIGINAL_TORCH_DEVICE_TYPE is: {_ORIGINAL_TORCH_DEVICE_TYPE}")
    log_info(f"type of _ORIGINAL_TORCH_DEVICE_TYPE is: {type(_ORIGINAL_TORCH_DEVICE_TYPE)}")
    
    if default_device == 'cuda':
        # For CUDA, call the original function
        torch.cuda.reset_peak_memory_stats()
    elif default_device == 'mps':
        # For MPS, we don't track peak memory separately
        pass
    else:
        pass

@auto_log()
def _get_mps_event_class(default_device):
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
        def __new__(cls, *args, **kwargs):
            # Bypass instantiation of a dummy base class by not calling the parent __new__
            return object.__new__(cls)
        
        @auto_log()
        def __init__(self, enable_timing=False, blocking=False, interprocess=False, device=None):
            # Do not call super().__init__() to avoid errors from a dummy base.
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
            if not self._recorded:
                return self
            if default_device == 'mps':
                import torch.mps
                torch.mps.synchronize()
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
            return (end_time - start_time) * 1000.0
        
        @auto_log()
        def synchronize(self):
            if not self._recorded:
                return self
            if default_device == 'mps':
                import torch.mps
                torch.mps.synchronize()
            return self
        
        @auto_log()
        def __del__(self):
            if hasattr(self, '_is_destroyed') and not self._is_destroyed:
                self._is_destroyed = True

    return MPSEvent

@auto_log()
def mock_cuda_event(default_device, *args, **kwargs):
    """Create a CUDA event. If on CUDA, delegate to torch.cuda.Event;
       otherwise, use our MPSEvent for MPS or CPU."""
    if default_device == 'cuda':
        return torch.cuda.Event(*args, **kwargs)
    # For non-CUDA, return an instance of our MPSEvent.
    MPSEvent = _get_mps_event_class(default_device)
    return MPSEvent(*args, **kwargs)

@auto_log()
def mock_cuda_stream(default_device, stream=None):
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
        def __exit__(self, exc_type, exc_val, traceback):
            if self.stream is not None and hasattr(self.stream, '__exit__'):
                return self.stream.__exit__(exc_type, exc_val, traceback)
            return False
    return StreamContext(stream)

@auto_log()
def mock_cuda_current_stream(default_device, device=None):
    return mock_cuda_stream_class(default_device, device=device)

@auto_log()
def mock_cuda_default_stream(default_device, device=None):
    return mock_cuda_stream_class(default_device, device=device)

@auto_log()
def mock_cuda_function_stub(default_device, *args, **kwargs):
    pass

@auto_log()
def tensor_creation_wrapper(original_func, default_device):
    def wrapped_func(*args, **kwargs):
        # If a device is explicitly provided:
        if 'device' in kwargs and kwargs['device'] is not None:
            device_arg = kwargs['device']
            # Convert string specs into a standard form:
            if isinstance(device_arg, str):
                requested_device = device_arg.split(':')[0]
                # If the requested device doesn't match the system default:
                if requested_device != default_device:
                    log_info(f"WARNING: Requested device '{requested_device}' is not available; redirecting to '{default_device}'")
                    kwargs['device'] = default_device
            # If it's a torch.device object:
            elif hasattr(device_arg, 'type'):
                requested_device = device_arg.type
                if requested_device != default_device:
                    log_info(f"WARNING: Requested device '{requested_device}' is not available; redirecting to '{default_device}'")
                    kwargs['device'] = default_device
        else:
            # No device specified; use the system's default.
            kwargs['device'] = default_device
            log_info(f"Using default device '{default_device}' for tensor creation")
        return original_func(*args, **kwargs)
    return wrapped_func
    