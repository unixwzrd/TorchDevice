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
from .modules.device_detection import get_default_device, redirect_device_type, _ORIGINAL_TORCH_DEVICE_TYPE

# Import the tensor creation wrapper from the CUDA mocks module.
from .modules.cuda_redirects import tensor_creation_wrapper

# Save original torch functions
_original_torch_device = torch.device


class TorchDevice:
    _default_device = None
    _lock = threading.RLock()
    _cpu_override = False

    @auto_log()
    def __init__(self, device_type: str = None, device_index: int = None):
        with self._lock:
            if self._default_device is None:
                self.__class__._default_device = get_default_device()
            if isinstance(device_type, str):
                if ':' in device_type:
                    device_type, index = device_type.split(':')
                    device_index = int(index)
                else:
                    device_index = 0 if device_index is None else device_index
                device_type = redirect_device_type(device_type, self._cpu_override)
                device_str = f"{device_type}:{device_index}"
                log_info(f"Creating torch.device('{device_str}')")
                self.device = _original_torch_device(device_str)

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
    def torch_device_replacement(cls, *args, **kwargs):
        """
        Drop-in replacement for torch.device() with device redirection and CPU override toggle.
        • No arguments → returns default device.
        • 'cpu:-1' → toggles CPU override.
        • Redirects non-CPU devices to available hardware.
        • Preserves extra args and kwargs.
        """
        # No arguments → return default device
        if not args and not kwargs:
            with cls._lock:
                if cls._default_device is None:
                    cls._default_device = get_default_device()
                default = cls._default_device
                if default.lower() == "cpu":
                    return _original_torch_device(default)
                return _original_torch_device(default, 0)

        # If first argument is torch.device, return as-is
        if args and isinstance(args[0], _ORIGINAL_TORCH_DEVICE_TYPE):
            return args[0]

        # If first argument is string device spec, parse and modify
        if args and isinstance(args[0], str):
            device_spec = args[0]
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

            with cls._lock:
                if cls._default_device is None:
                    cls._default_device = get_default_device()

                # CPU toggle logic
                if device_type == "cpu" and device_index == -1:
                    if cls._cpu_override:
                        # Toggle OFF
                        cls._cpu_override = False
                        device_type = cls._default_device
                        device_index = None
                    else:
                        # Toggle ON
                        cls._cpu_override = True
                        device_type = "cpu"
                        device_index = 0

                # Apply redirection if no CPU override
                if not cls._cpu_override:
                    device_type = redirect_device_type(device_type)

                # Reassemble args
                new_arg = device_type
                if device_index is not None:
                    new_arg = f"{device_type}:{device_index}"

                args = (new_arg,) + args[1:]  # Replace first arg

        # Pass everything through to original torch.device
        rvalue = _original_torch_device(*args, **kwargs)
        return rvalue


    @classmethod
    @auto_log()
    def apply_patches(cls):
        """
        Apply all patches to the torch API.
        """
        from .modules.device_detection import get_default_device

        # Save the original torch.device constructor.
        _original_torch_device = torch.device

        @auto_log()
        def patched_torch_device(*args, **kwargs):
            """
            A patched version of torch.device.
            • If called with no arguments, returns the default device (for non‑CPU, forcing index 0).
            • If called with arguments that look like a device specification (string or torch.device),
            it routes them through our torch_device_replacement.
            • Otherwise, it passes the arguments directly to the original constructor.
            """
            # _original_torch_device is assumed to be saved already.
            if not args and not kwargs:
                default = get_default_device()
                log_info(f"torch.device() called with no arguments; using default device '{default}'")
                if default.lower() == "cpu":
                    device = _original_torch_device(default)
                else:
                    device = _original_torch_device(default, 0)
            # If the first argument looks like a device specification, route through our replacement.
            if args and (isinstance(args[0], str) or isinstance(args[0], _ORIGINAL_TORCH_DEVICE_TYPE)):
                device = TorchDevice.torch_device_replacement(*args, **kwargs)
                log_info(f"torch.device() called with arguments; replacement device '{device}' patched_torch_device")
                return device
            return _original_torch_device(*args, **kwargs)

        # Patch torch.device globally.
        torch.device = patched_torch_device

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
                setattr(torch, func_name, tensor_creation_wrapper(original_func, cls._default_device))

        # Patch torch.cuda functions using the mocks.
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
