"""
TorchDevice - Transparent PyTorch Device Redirection

This module enables seamless code portability between NVIDIA CUDA, Apple Silicon (MPS),
and CPU hardware for PyTorch applications. It intercepts PyTorch calls related to GPU
hardware, allowing developers to write code that works across different hardware
without modification.

Key features:
- Automatic device redirection based on available hardware
- CPU override capability using 'cpu:-1' device specification
- Detailed logging for debugging and migration assistance

Usage:
    from TorchDevice import TorchDevice
    # Use TorchDevice to get a torch.device object reflecting redirection logic.
    device_obj = TorchDevice(device_type='cuda')
    print(device_obj)
"""
import threading
import torch
from .modules.TDLogger import log_info, auto_log
from .modules.device_detection import get_default_device, get_default_device, redirect_device_type, _ORIGINAL_TORCH_DEVICE_TYPE

# Import the tensor creation wrapper from the CUDA mocks module.
from .modules.cuda_redirects import tensor_creation_wrapper


class TorchDevice:
    _default_device = None
    _lock = threading.Lock()
    _cpu_override = False
    _natural_default = None  # Store the natural default when CPU override is activated


    @auto_log()
    def __init__(self, device_type: str = None, device_index: int = None):
        with self._lock:
            if self._default_device is None:
                self.__class__._default_device = _CACHED_DEFAULT_DEVICE
            if device_type is None:
                device_type = self._default_device
            if isinstance(device_type, str):
                if ':' in device_type:
                    device_type, index = device_type.split(':')
                    device_index = int(index)
                else:
                    device_index = 0 if device_index is None else device_index
                device_type = redirect_device_type(device_type, self._cpu_override)
                device_str = f"{device_type}:{device_index}"
                log_info(f"Creating torch.device('{device_str}')")
                self.device = torch.device(device_str)
            else:
                self.device = torch.device(device_type)

    @auto_log()
    def __repr__(self):
        return repr(self.device)

    @auto_log()
    def __str__(self):
        return str(self.device)

    @auto_log()
    def __getattr__(self, attr):
        return getattr(self.device, attr)

    @classmethod
    @auto_log()
    def toggle_cpu_override(cls, device_spec: str) -> torch.device:
        """
        If device_spec is "cpu:-1", toggle the CPU override.
        If not currently overridden, store the natural default and force CPU.
        If already overridden, revert to the natural default.
        """
        if device_spec == "cpu:-1":
            if not cls._cpu_override:
                # Activate override: store the natural default and force CPU.
                cls._natural_default = cls._default_device
                cls._cpu_override = True
                cls._default_device = "cpu"
                log_info("Activating CPU override: forcing device to 'cpu'")
            else:
                # Revert the override: restore the natural default.
                cls._cpu_override = False
                if cls._natural_default is not None:
                    cls._default_device = cls._natural_default
                    log_info(f"Reverting CPU override: restoring natural default '{cls._natural_default}'")
                    cls._natural_default = None
            return torch.device(cls._default_device)
        else:
            # If the device_spec is not the special toggle, just return it as a torch.device.
            return torch.device(device_spec)

    @classmethod
    @auto_log()
    def torch_device_replacement(cls, device_type="", device_index=None):
        """
        Replacement for torch.device to handle redirection logic.
        """
        if not device_type:
            with cls._lock:
                if cls._default_device is None:
                    cls._default_device = _CACHED_DEFAULT_DEVICE
                device_type = cls._default_device
            return torch.device(device_type)

        def set_cpu_override():
            with cls._lock:
                cls._default_device = 'cpu'
                cls._cpu_override = True
                log_info("CPU override is set, using CPU as default device")
            return torch.device('cpu')

        if isinstance(device_type, _ORIGINAL_TORCH_DEVICE_TYPE):
            name = device_type.type
            index = device_type.index
            if name == 'cpu' and index == -1:
                return set_cpu_override()
            if name == 'mps' and torch.backends.mps.is_available():
                return torch.device('mps')
            redirected = redirect_device_type(name, cls._cpu_override)
            if redirected != name:
                return torch.device(redirected)
            return device_type

        if isinstance(device_type, str):
            if ':' in device_type:
                name, index = device_type.split(":", 1)
                if name == 'cpu' and index == '-1':
                    return set_cpu_override()
                if name == 'mps' and torch.backends.mps.is_available():
                    return torch.device('mps')
                redirected = redirect_device_type(name, cls._cpu_override)
                device_str = f"{redirected}:{index}" if redirected != name else device_type
            else:
                if device_type == 'mps' and torch.backends.mps.is_available():
                    return torch.device('mps')
                redirected = redirect_device_type(device_type, cls._cpu_override)
                device_str = redirected if redirected != device_type else device_type
            return torch.device(device_str)

        with cls._lock:
            if cls._default_device is None:
                cls._default_device = _CACHED_DEFAULT_DEVICE
        if device_type == 'cpu' and device_index == -1:
            return set_cpu_override()
        if device_type == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        if isinstance(device_type, str):
            redirected = redirect_device_type(device_type, cls._cpu_override)
            if redirected != device_type:
                device_type = redirected
        if device_index is not None:
            try:
                device_index = int(device_index)
            except ValueError:
                pass
            return torch.device(device_type, device_index)
        return torch.device(device_type)

    @classmethod
    @auto_log()
    def apply_patches(cls):
        """
        Apply all patches to the torch API.
        """
        from .modules.patching import apply_basic_patches
        # Cache the default device.
        default = get_default_device()  # This call caches the result internally.
        cls._default_device = default
        apply_basic_patches()
        
        # Patch tensor creation functions (if no explicit device is provided, use the cached default).
        tensor_creation_functions = [
            'tensor', 'zeros', 'ones', 'empty', 'randn', 'rand', 'randint',
            'arange', 'linspace', 'logspace'
        ]
        for func_name in tensor_creation_functions:
            if hasattr(torch, func_name):
                original_func = getattr(torch, func_name)
                patched_func = tensor_creation_wrapper(original_func, cls._default_device)
                setattr(torch, func_name, patched_func)
        
        # Patch CUDA functions using our mocks.
        torch.cuda.is_available = lambda: cls._default_device in ['cuda', 'mps']
        torch.cuda.device_count = lambda: 1 if cls._default_device == 'mps' else (torch.cuda.device_count() if cls._default_device == 'cuda' else 0)
        
        from .modules.cuda_redirects import (
            mock_cuda_get_device_properties, mock_cuda_empty_cache,
            mock_cuda_synchronize, mock_cuda_current_device, mock_cuda_set_device,
            mock_cuda_get_device_name, mock_cuda_get_device_capability,
            mock_cuda_memory_allocated, mock_cuda_memory_reserved,
            mock_cuda_max_memory_allocated, mock_cuda_max_memory_reserved,
            mock_cuda_memory_stats, mock_cuda_memory_snapshot, mock_cuda_memory_summary,
            mock_cuda_is_initialized, mock_cuda_get_arch_list, mock_cuda_is_built,
            mock_cuda_device_context, mock_cuda_stream_class, mock_cuda_stream,
            mock_cuda_current_stream, mock_cuda_default_stream, _get_mps_event_class
        )
        
        torch.cuda.get_device_properties = lambda device: mock_cuda_get_device_properties(cls._default_device, device)
        torch.cuda.empty_cache = lambda: mock_cuda_empty_cache(cls._default_device)
        torch.cuda.synchronize = lambda device=None: mock_cuda_synchronize(cls._default_device, device)
        torch.cuda.current_device = lambda: mock_cuda_current_device(cls._default_device)
        torch.cuda.set_device = lambda device: mock_cuda_set_device(cls._default_device, device)
        torch.cuda.get_device_name = lambda device=None: mock_cuda_get_device_name(cls._default_device, device)
        torch.cuda.get_device_capability = lambda device=None: mock_cuda_get_device_capability(cls._default_device, device)
        torch.cuda.memory_allocated = lambda device=None: mock_cuda_memory_allocated(cls._default_device, device)
        torch.cuda.memory_reserved = lambda device=None: mock_cuda_memory_reserved(cls._default_device, device)
        torch.cuda.max_memory_allocated = lambda device=None: mock_cuda_max_memory_allocated(cls._default_device, device)
        torch.cuda.max_memory_reserved = lambda device=None: mock_cuda_max_memory_reserved(cls._default_device, device)
        torch.cuda.memory_stats = lambda device=None: mock_cuda_memory_stats(cls._default_device, device)
        torch.cuda.memory_snapshot = lambda: mock_cuda_memory_snapshot(cls._default_device)
        torch.cuda.memory_summary = lambda device=None, abbreviated=False: mock_cuda_memory_summary(cls._default_device, device, abbreviated)
        torch.cuda.is_initialized = lambda: mock_cuda_is_initialized(cls._default_device)
        torch.cuda.get_arch_list = lambda: mock_cuda_get_arch_list(cls._default_device)
        torch.backends.cuda.is_built = lambda: mock_cuda_is_built(cls._default_device)
        
        # Do not override torch.cuda.device, so its type remains unchanged.
        
        torch.cuda.Stream = lambda *args, **kwargs: mock_cuda_stream_class(cls._default_device, *args, **kwargs)
        torch.cuda.stream = lambda stream=None: mock_cuda_stream(cls._default_device, stream)
        torch.cuda.current_stream = lambda device=None: mock_cuda_current_stream(cls._default_device, device)
        torch.cuda.default_stream = lambda device=None: mock_cuda_default_stream(cls._default_device, device)
        
        # Override torch.cuda.Event with our proper event class.
        torch.cuda.Event = _get_mps_event_class(cls._default_device)
        
        # Set unsupported functions to no-ops.
        unsupported_functions = [
            'set_stream', 'mem_get_info', 'reset_accumulated_memory_stats',
            'reset_max_memory_allocated', 'reset_max_memory_cached',
            'caching_allocator_alloc', 'caching_allocator_delete',
            'get_allocator_backend', 'change_current_allocator', 'nvtx',
            'jiterator', 'graph', 'CUDAGraph', 'make_graphed_callables',
            'is_current_stream_capturing', 'graph_pool_handle', 'can_device_access_peer',
            'comm', 'get_gencode_flags', 'current_blas_handle', 'memory_usage',
            'utilization', 'temperature', 'power_draw', 'clock_rate',
            'set_sync_debug_mode', 'get_sync_debug_mode', 'list_gpu_processes',
            'seed', 'seed_all', 'manual_seed', 'manual_seed_all',
            'get_rng_state', 'get_rng_state_all', 'set_rng_state',
            'set_rng_state_all', 'initial_seed',
        ]
        for func_name in unsupported_functions:
            if hasattr(torch.cuda, func_name):
                setattr(torch.cuda, func_name, lambda *args, **kwargs: None)

def initialize_torchdevice():
    """
    Initialize TorchDevice:
        1. Set the global default device.
        2. Apply all patches to the torch API.
        3. Log the initialization.
    """
    if TorchDevice._default_device is None:
        TorchDevice._default_device = get_default_device()
    TorchDevice.apply_patches()
    log_info(f"TorchDevice initialization complete. Default device: {get_default_device()}")
