"""
TorchDevice CUDA Operations Module
------------------------------
CUDA-specific operations and patches.
"""

import torch
import types
import sys
from typing import Optional, Any, Callable, List, Dict, Union
from ...core.logger import log_info, log_warning, log_error, auto_log

try:
    import psutil
except ImportError:
    psutil = None # type: ignore
    log_info("psutil library not found. Memory simulation will be basic.")

# --- Canonical Module References --- 
# These must be defined early, as PyTorch might import and cache them internally.
import inspect # For type checking if needed, though explicit overwrite is planned
from ...core import hardware_info # For context-aware stubs

_cuda_module_ref = sys.modules.get('torch.cuda')
_backends_cuda_module_ref = sys.modules.get('torch.backends.cuda')

# Ensure torch.cuda exists as a module object and is assigned to torch.cuda
if not _cuda_module_ref:
    log_warning("TorchDevice (module load ops.device.cuda): torch.cuda not found in sys.modules. Creating a mock module.")
    _cuda_module_ref = types.ModuleType('torch.cuda')
    setattr(_cuda_module_ref, '__module_origin__', 'TorchDevice_mock') # Mark our mock
    torch.cuda = _cuda_module_ref # Assign the mock module to torch.cuda
    sys.modules['torch.cuda'] = _cuda_module_ref # Ensure global import visibility
    log_info("TorchDevice (ops.device.cuda module load): Created mock torch.cuda and updated torch.cuda (ID: %s) and sys.modules['torch.cuda'] (ID: %s).", id(torch.cuda), id(sys.modules['torch.cuda']))

    # Also create mock torch.backends.cuda if it's missing or if torch.cuda was missing
    # Check if torch.backends exists first
    if not hasattr(torch, 'backends') or not hasattr(torch.backends, 'cuda') or not __is_torch_actually_compiled_with_cuda_value:
        log_info("TorchDevice (ops.device.cuda module load): Creating mock torch.backends.cuda.")
        _backends_cuda_module_ref = types.ModuleType('torch.backends.cuda')
        setattr(_backends_cuda_module_ref, '__module_origin__', 'TorchDevice_mock') # Mark our mock
        if not hasattr(torch, 'backends'):
            # If torch.backends itself doesn't exist, create it as a mock module
            log_info("TorchDevice (ops.device.cuda module load): torch.backends does not exist, creating mock torch.backends module.")
            torch.backends = types.ModuleType('torch.backends')
            setattr(torch.backends, '__module_origin__', 'TorchDevice_mock_parent')
            sys.modules['torch.backends'] = torch.backends
        torch.backends.cuda = _backends_cuda_module_ref
        sys.modules['torch.backends.cuda'] = _backends_cuda_module_ref
        log_info("TorchDevice (ops.device.cuda module load): Created mock torch.backends.cuda and updated torch.backends.cuda (ID: %s) and sys.modules['torch.backends.cuda'] (ID: %s).", id(torch.backends.cuda), id(sys.modules['torch.backends.cuda']))
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'cuda'):
        _backends_cuda_module_ref = torch.backends.cuda_module_ref
    log_info("TorchDevice (module load ops.device.cuda): Mock torch.backends.cuda module created and assigned.")
elif not hasattr(torch.backends, 'cuda'):
    log_warning("TorchDevice (module load ops.device.cuda): sys.modules['torch.backends.cuda'] exists but torch.backends.cuda attribute is missing. Assigning.")
    torch.backends.cuda = _backends_cuda_module_ref

# Ensure torch.backends.cuda exists as a module object and is assigned to torch.backends.cuda
if not _backends_cuda_module_ref:
    log_warning("TorchDevice (module load ops.device.cuda): torch.backends.cuda not found in sys.modules. Creating mock module.")
    _backends_cuda_module_ref = types.ModuleType('torch.backends.cuda')
    sys.modules['torch.backends.cuda'] = _backends_cuda_module_ref
    if not hasattr(torch.backends, 'cuda'):
        torch.backends.cuda = _backends_cuda_module_ref
    log_info("TorchDevice (module load ops.device.cuda): Mock torch.backends.cuda module created and assigned.")
elif not hasattr(torch.backends, 'cuda'):
    log_warning("TorchDevice (module load ops.device.cuda): sys.modules['torch.backends.cuda'] exists but torch.backends.cuda attribute is missing. Assigning.")
    torch.backends.cuda = _backends_cuda_module_ref

# --- Globals to store original PyTorch CUDA functions and compilation status ---
_original_lazy_init = None
_original_is_initialized = None

# Store original functions using the canonical module references
t_Tensor_cuda = getattr(torch.Tensor, 'cuda', None)
t_nn_Module_cuda = getattr(torch.nn.Module, 'cuda', None)

# Store originals from _cuda_module_ref if it exists
t_cuda_current_device = getattr(_cuda_module_ref, 'current_device', None) if _cuda_module_ref else None
t_cuda_device = getattr(_cuda_module_ref, 'device', None) if _cuda_module_ref else None # Context manager
t_cuda_device_count = getattr(_cuda_module_ref, 'device_count', None) if _cuda_module_ref else None
t_cuda_empty_cache = getattr(_cuda_module_ref, 'empty_cache', None) if _cuda_module_ref else None
t_cuda_get_arch_list = getattr(_cuda_module_ref, 'get_arch_list', None) if _cuda_module_ref else None
t_cuda_get_device_capability = getattr(_cuda_module_ref, 'get_device_capability', None) if _cuda_module_ref else None
t_cuda_get_device_name = getattr(_cuda_module_ref, 'get_device_name', None) if _cuda_module_ref else None
t_cuda_get_device_properties = getattr(_cuda_module_ref, 'get_device_properties', None) if _cuda_module_ref else None
# _original_is_available will be t_cuda_is_available, handled with _original_is_initialized below
# _original_is_initialized is handled specifically later with _lazy_init
t_cuda_set_device = getattr(_cuda_module_ref, 'set_device', None) if _cuda_module_ref else None
t_cuda_synchronize = getattr(_cuda_module_ref, 'synchronize', None) if _cuda_module_ref else None

# AMP hooks - store originals from _cuda_module_ref.amp
t_cuda_amp_autocast = None
t_cuda_amp_GradScaler = None
if _cuda_module_ref and hasattr(_cuda_module_ref, 'amp'):
    _cuda_amp_module_ref = getattr(_cuda_module_ref, 'amp')
    t_cuda_amp_autocast = getattr(_cuda_amp_module_ref, 'autocast', None)
    if hasattr(_cuda_amp_module_ref, 'GradScaler'):
        t_cuda_amp_GradScaler = getattr(_cuda_amp_module_ref, 'GradScaler', None)

# Determine once if PyTorch is actually compiled with CUDA support
__is_torch_actually_compiled_with_cuda_value = hasattr(torch._C, '_cuda_getDeviceCount')

def _is_torch_actually_compiled_with_cuda() -> bool:
    """Return True if PyTorch was compiled with CUDA support, False otherwise."""
    return __is_torch_actually_compiled_with_cuda_value

log_info("TorchDevice (module load ops.device.cuda): Is PyTorch actually compiled with CUDA (via torch._C._cuda_getDeviceCount): %s", __is_torch_actually_compiled_with_cuda_value)

if _cuda_module_ref:
    _original_lazy_init = getattr(_cuda_module_ref, '_lazy_init', None)
    log_info("TorchDevice (module load): Stored original _lazy_init from _cuda_module_ref: %s", _original_lazy_init is not None)
    _original_is_initialized = getattr(_cuda_module_ref, 'is_initialized', None)
    log_info("TorchDevice (module load): Stored original is_initialized from _cuda_module_ref: %s", _original_is_initialized is not None)
else:
    # This case was handled by the log_warning above, but set to None for safety
    _original_lazy_init = None
    _original_is_initialized = None
    # log_warning already issued if _cuda_module_ref is None

# Tensor movement functions are now in core.tensors
# Use the consolidated implementations from core.tensors module
# instead of duplicating the functionality here

# CUDA backend function wrappers
@auto_log()
def backends_cuda_is_built_replacement():
    """Simulate that CUDA is built for all platforms to support CUDA code.
    
    According to the TorchDevice behavior documentation, we should simulate
    CUDA features on all platforms to ensure CUDA code can run without modification.
    """
    # Always return True as part of CUDA feature simulation layer
    # This ensures CUDA code runs without modification on MPS and CPU
    return True

@auto_log()
def cuda_is_available_replacement():
    """Simulate that CUDA is available for all platforms to support CUDA code.
    
    According to the TorchDevice behavior documentation, we should simulate
    CUDA features on all platforms to ensure CUDA code can run without modification.
    """
    # Always return True as part of CUDA feature simulation layer
    # This allows CUDA code to believe CUDA is available on MPS and CPU
    return True

@auto_log()
def cuda_set_device_replacement(device):
    """Set device replacement that doesn't fail on non-CUDA devices."""
    # Log but don't actually try to set CUDA device on non-CUDA hardware
    pass

@auto_log()
def cuda_get_device_properties_replacement(device):
    """Return simulated device properties for non-CUDA devices."""
    # Create a minimal property object that won't cause errors
    class DeviceProperties:
        def __init__(self):
            self.name = "CUDA (Simulated)"
            self.major = 7
            self.minor = 0
            self.total_memory = 8 * (1024**3)  # 8GB
            self.multi_processor_count = 8
            self.max_threads_per_block = 1024
            self.max_grid_size = (2147483647, 65535, 65535)
            self.max_block_dim = (1024, 1024, 64)
    
    return DeviceProperties()

# --- Simulated Memory Management Functions ---
@auto_log()
def cuda_empty_cache_replacement():
    """Simulated torch.cuda.empty_cache()."""
    log_info("Simulated torch.cuda.empty_cache() called.")
    # No-op for simulation

@auto_log()
def cuda_memory_allocated_replacement(device=None):
    """Simulated torch.cuda.memory_allocated()."""
    log_info("Simulated torch.cuda.memory_allocated(device=%s) called. Returning 0.", device)
    return 0

@auto_log()
def cuda_max_memory_allocated_replacement(device=None):
    """Simulated torch.cuda.max_memory_allocated()."""
    log_info("Simulated torch.cuda.max_memory_allocated(device=%s) called. Returning 0.", device)
    return 0

@auto_log()
def cuda_reset_peak_memory_stats_replacement(device=None):
    """Simulated torch.cuda.reset_peak_memory_stats()."""
    log_info("Simulated torch.cuda.reset_peak_memory_stats(device=%s) called.", device)
    # No-op for simulation

@auto_log()
def cuda_memory_reserved_replacement(device=None):
    """Simulated torch.cuda.memory_reserved()."""
    log_info("Simulated torch.cuda.memory_reserved(device=%s) called. Returning 0.", device)
    return 0

@auto_log()
def cuda_max_memory_reserved_replacement(device=None):
    """Simulated torch.cuda.max_memory_reserved()."""
    log_info("Simulated torch.cuda.max_memory_reserved(device=%s) called. Returning 0.", device)
    return 0

@auto_log()
def cuda_memory_stats_replacement(device=None):
    """Simulated torch.cuda.memory_stats(). Returns a dict with common keys and zero values."""
    log_info("Simulated torch.cuda.memory_stats(device=%s) called. Returning placeholder stats.", device)
    # Return a dictionary with common keys expected by tests, with 0 values
    # Based on torch.cuda.memory_stats() output structure
    return {
        'active.all.current': 0,
        'active.all.peak': 0,
        'active_bytes.all.current': 0,
        'active_bytes.all.peak': 0,
        'allocated_bytes.all.current': 0,
        'allocated_bytes.all.peak': 0,
        'inactive_split.all.current': 0,
        'inactive_split.all.peak': 0,
        'inactive_split_bytes.all.current': 0,
        'inactive_split_bytes.all.peak': 0,
        'reserved_bytes.all.current': 0,
        'reserved_bytes.all.peak': 0,
        # Add other common keys if tests require them
    }

_cuda_module_ref = sys.modules.get('torch.cuda')
_backends_cuda_module_ref = sys.modules.get('torch.backends.cuda')

if _cuda_module_ref:
    _original_lazy_init = getattr(_cuda_module_ref, '_lazy_init', None)
    log_info("TorchDevice (module load): Stored original _lazy_init from _cuda_module_ref: %s", _original_lazy_init is not None)
    _original_is_initialized = getattr(_cuda_module_ref, 'is_initialized', None)
    log_info("TorchDevice (module load): Stored original is_initialized from _cuda_module_ref: %s", _original_is_initialized is not None)
else:
    _original_lazy_init = None
    _original_is_initialized = None
    log_warning("TorchDevice (module load): _cuda_module_ref is None. Cannot store original CUDA init functions.")

@auto_log()
def _universal_lazy_init_replacement():
    """
    Universal replacement for torch.cuda._lazy_init.
    Ensures that _initialized is set to True if CUDA is not compiled,
    otherwise calls the original _lazy_init.
    """
    # Directly use the global value and log it for clarity
    is_compiled = __is_torch_actually_compiled_with_cuda_value
    log_info("TorchDevice (_universal_lazy_init_replacement): Entered. PyTorch compiled with CUDA (observed value): %s", is_compiled)
    
    if not is_compiled:
        if _cuda_module_ref:
            # Ensure _initialized is set on the actual torch.cuda module object
            setattr(_cuda_module_ref, '_initialized', True)
            log_info("TorchDevice (_universal_lazy_init): CUDA not compiled, ensured _cuda_module_ref._initialized = True")
        return # IMPORTANT: Return here to prevent calling original_lazy_init
    
    # If execution reaches here, it means is_compiled is True
    if _original_lazy_init:
        log_info("TorchDevice (_universal_lazy_init): CUDA compiled, calling original _lazy_init.")
        return _original_lazy_init()
    else:
        # Fallback if original_lazy_init is somehow None despite CUDA being compiled
        if _cuda_module_ref:
            setattr(_cuda_module_ref, '_initialized', True)
        log_warning("TorchDevice (_universal_lazy_init): CUDA compiled, but _original_lazy_init is None. Setting _initialized to True as fallback.")

@auto_log()
def _universal_is_initialized_replacement():
    """
    Universal replacement for torch.cuda.is_initialized.
    If CUDA is not compiled, returns the state of _cuda_module_ref._initialized.
    Otherwise, calls the original is_initialized.
    """
    is_compiled = __is_torch_actually_compiled_with_cuda_value # Use the global boolean
    log_info("TorchDevice (_universal_is_initialized_replacement): Entered. PyTorch compiled with CUDA (observed value): %s", is_compiled)

    if not is_compiled:
        initialized_status = getattr(_cuda_module_ref, '_initialized', False)
        log_info("TorchDevice (_universal_is_initialized): CUDA not compiled, returning _cuda_module_ref._initialized: %s", initialized_status)
        return initialized_status
    
    # If execution reaches here, it means is_compiled is True
    if _original_is_initialized:
        log_info("TorchDevice (_universal_is_initialized): CUDA compiled, calling original is_initialized.")
        return _original_is_initialized()
    else:
        # Fallback if original_is_initialized is somehow None despite CUDA being compiled
        fallback_status = getattr(_cuda_module_ref, '_initialized', True) # Default to True if original is missing
        log_warning(f"TorchDevice (_universal_is_initialized): CUDA compiled, but _original_is_initialized is None. Returning fallback _initialized status: {fallback_status}.")
        return fallback_status

def apply_patches():
    """
    Apply patches for CUDA-specific functions.
    
    This function replaces critical CUDA functionality with compatible versions that:
    1. Run CUDA code on MPS hardware transparently (if PyTorch is CUDA-compiled)
    2. Provide comprehensive, inert stubs if PyTorch is NOT CUDA-compiled.
    3. Respect CPU override for testing.
    4. Properly log device redirections.
    5. Never fail due to missing CUDA hardware.
    
    The behavior of patches differs based on whether PyTorch itself was compiled
    with CUDA support, and whether the current execution device is CUDA.
    """
    from TorchDevice.core.device import DeviceManager # Local import to break circular dependency

    current_device = DeviceManager.get_default_device()
    is_cuda_device = current_device and current_device.type == 'cuda'

    if is_cuda_device:
        log_info("TorchDevice (apply_patches ops.device.cuda): Skipping CUDA patches, running on actual CUDA hardware.")
        return

    log_info("TorchDevice (apply_patches ops.device.cuda): Applying CUDA patches on non-CUDA hardware (%s).", current_device.type if current_device else 'unknown')

    if not _cuda_module_ref:
        log_error("TorchDevice (apply_patches ops.device.cuda): _cuda_module_ref is None. Cannot apply CUDA patches.")
        return

    # Universal patches for initialization state, availability, and build status
    # These are applied regardless of whether PyTorch was compiled with CUDA, as long as we're on non-CUDA hardware.
    # Ensure these attributes exist before patching, or create them if necessary.
    if not hasattr(_cuda_module_ref, '_lazy_init'):
        log_warning("TorchDevice (apply_patches ops.device.cuda): _cuda_module_ref has no _lazy_init. Creating stub.")
    _cuda_module_ref._lazy_init = _universal_lazy_init_replacement
    
    if not hasattr(_cuda_module_ref, 'is_initialized'):
        log_warning("TorchDevice (apply_patches ops.device.cuda): _cuda_module_ref has no is_initialized. Creating stub.")
    _cuda_module_ref.is_initialized = _universal_is_initialized_replacement
    
    if not hasattr(_cuda_module_ref, 'is_available'):
        log_warning("TorchDevice (apply_patches ops.device.cuda): _cuda_module_ref has no is_available. Creating stub.")
    _cuda_module_ref.is_available = cuda_is_available_replacement # Returns True
    
    if _backends_cuda_module_ref:
        if not hasattr(_backends_cuda_module_ref, 'is_built'):
            log_warning("TorchDevice (apply_patches ops.device.cuda): torch.backends.cuda has no is_built. Creating stub.")
        _backends_cuda_module_ref.is_built = backends_cuda_is_built_replacement # Returns True
    else:
        log_warning("TorchDevice (apply_patches ops.device.cuda): torch.backends.cuda module not found. Cannot patch is_built.")

    # Call _lazy_init here to ensure _initialized is set correctly by our replacement if needed
    _cuda_module_ref._lazy_init()
    log_info("TorchDevice (apply_patches ops.device.cuda): Called patched _lazy_init. _cuda_module_ref._initialized is now: %s", getattr(_cuda_module_ref, '_initialized', 'Not Set'))

    if not _is_torch_actually_compiled_with_cuda():
        log_info("TorchDevice (apply_patches ops.device.cuda): PyTorch NOT compiled with CUDA. Applying comprehensive stubs.")
        # Ensure _initialized is True for stub mode, even if _lazy_init didn't create/set it.
        if not hasattr(_cuda_module_ref, '_initialized'):
            _cuda_module_ref._initialized = True
        else:
            _cuda_module_ref._initialized = True # type: ignore[attr-defined]

        # Comprehensive stubs for torch.cuda attributes based on test_cuda_stubs.py
        # These are explicitly set to ensure they exist and are callable with expected inert behavior.

        # Define stubs as local functions to get stable IDs for logging
        def _stub_current_device_non_cuda(): return 0
        def _stub_device_count_non_cuda(): return 1 if hardware_info.is_native_mps_available() else 0

        stubs_for_non_cuda_build = {
            # Submodules (create as mock modules)
            'nvtx': types.ModuleType('torch.cuda.nvtx'),
            'jiterator': types.ModuleType('torch.cuda.jiterator'),
            'graph': types.ModuleType('torch.cuda.graph'),
            'comm': types.ModuleType('torch.cuda.comm'),
            # Classes (create as mock classes)
            'CUDAGraph': type('CUDAGraph', (object,), {'__init__': lambda self, *a, **kw: None}),
            'device': type('device', (object,), { # Stub for torch.cuda.device context manager
                '__init__': lambda self, idx: None,
                '__enter__': lambda self: None,
                '__exit__': lambda self, *args: None
            }),
            # Functions with specific return values for non-CUDA based on test_cuda_stubs.py
            'set_stream': lambda *a, **kw: None,
            'mem_get_info': lambda *a, **kw: (0, 0), # test expects tuple or None
            'reset_accumulated_memory_stats': lambda *a, **kw: None,
            'reset_max_memory_allocated': lambda *a, **kw: None,
            'reset_max_memory_cached': lambda *a, **kw: None,
            'caching_allocator_alloc': lambda *a, **kw: 0, # Test expects None, 0 is a safe stub for a pointer
            'caching_allocator_delete': lambda *a, **kw: None,
            'get_allocator_backend': lambda *a, **kw: "stubbed_backend_non_cuda", # Test expects None
            'change_current_allocator': lambda *a, **kw: None,
            'make_graphed_callables': lambda *a, **kw: [], # Test expects None
            'is_current_stream_capturing': lambda *a, **kw: False, # Test expects None
            'graph_pool_handle': lambda *a, **kw: None,
            'can_device_access_peer': lambda *a, **kw: False, # Test expects None
            'get_gencode_flags': lambda *a, **kw: "", # Test expects None
            'current_blas_handle': lambda *a, **kw: None,
            'memory_usage': lambda *a, **kw: 0, # Test expects None
            'utilization': lambda *a, **kw: 0,
            'temperature': lambda *a, **kw: 0,
            'power_draw': lambda *a, **kw: 0,
            'clock_rate': lambda *a, **kw: 0,
            'set_sync_debug_mode': lambda *a, **kw: None,
            'get_sync_debug_mode': lambda *a, **kw: 0,
            'list_gpu_processes': lambda *a, **kw: [],
            # General device functions (some use existing _replacement stubs)
            'device_count': _stub_device_count_non_cuda,
            'current_device': _stub_current_device_non_cuda,
            'get_device_name': lambda device=None: "MPS (Simulated by TorchDevice)" if hardware_info.is_native_mps_available() else "CUDA (Simulated by TorchDevice)",
            'get_device_capability': lambda device=None: (1, 0), # Generic capability for MPS/CPU simulation
            'get_device_properties': cuda_get_device_properties_replacement,
            'set_device': cuda_set_device_replacement,
            'empty_cache': cuda_empty_cache_replacement,
            'memory_allocated': cuda_memory_allocated_replacement,
            'max_memory_allocated': cuda_max_memory_allocated_replacement,
            'reset_peak_memory_stats': cuda_reset_peak_memory_stats_replacement,
            'memory_reserved': cuda_memory_reserved_replacement,
            'max_memory_reserved': cuda_max_memory_reserved_replacement,
            'memory_stats': cuda_memory_stats_replacement,
            'memory_summary': lambda device=None, abbreviated=False: "Simulated memory summary (TorchDevice)\nMemory Allocated: 0 MiB\nMemory Reserved: 0 MiB\nTotal Capacity: 0 MiB", # Stub for memory_summary
            'memory_snapshot': lambda: [],
            'get_arch_list': lambda: ['sm_70'] if hardware_info.is_native_mps_available() else [], # Simulate some arch for MPS
            # Stream and Event stubs (minimal, full patching by their modules)
            'Stream': type('Stream', (object,), {'__init__': lambda s, *a, **k: None, 'synchronize': lambda s: None, 'query': lambda s: True, 'record_event': lambda s, e=None: e if e else type('Event', (object,), {'record':lambda ev:None})(), 'wait_event': lambda s,e: None, 'wait_stream': lambda s,st: None, '__enter__': lambda s: s, '__exit__': lambda s,*a: None }),
            'Event': type('Event', (object,), {'__init__': lambda s, *a, **k: None, 'record': lambda s, st=None: None, 'synchronize': lambda s: None, 'query': lambda s: True, 'wait': lambda s, st=None: None, 'elapsed_time': lambda s,e: 0.0}),
            'current_stream': lambda device=None: None, # Test expects None
            'default_stream': lambda device=None: None, # Test expects None
            'stream': lambda stream=None: stubs_for_non_cuda_build['Stream'](stream) if stream is not None else stubs_for_non_cuda_build['Stream'](), # Factory for Stream context manager
            'ipc_collect': lambda: None, # Stub for ipc_collect
        }

        for name, stub_impl in stubs_for_non_cuda_build.items():
            setattr(_cuda_module_ref, name, stub_impl)
            if name == 'current_device':
                log_info("TorchDevice (apply_patches ops.device.cuda): Stubbed torch.cuda.current_device with ID: %s (local func ID: %s)", id(stub_impl), id(_stub_current_device_non_cuda))
            elif name == 'device_count':
                log_info("TorchDevice (apply_patches ops.device.cuda): Stubbed torch.cuda.device_count with ID: %s (local func ID: %s)", id(stub_impl), id(_stub_device_count_non_cuda))

        # AMP Stubs
        if not hasattr(_cuda_module_ref, 'amp') or not isinstance(getattr(_cuda_module_ref, 'amp'), types.ModuleType):
            _cuda_module_ref.amp = types.ModuleType('torch.cuda.amp')
            log_info("TorchDevice (apply_patches ops.device.cuda): Created torch.cuda.amp submodule for non-CUDA build.")
        _cuda_module_ref.amp.autocast = _cuda_amp_autocast_replacement
        _cuda_module_ref.amp.GradScaler = _GradScalerReplacement
        log_info("TorchDevice (apply_patches ops.device.cuda): Applied comprehensive stubs for non-CUDA build.")

    else: # PyTorch IS compiled with CUDA, but running on non-CUDA hardware (e.g., MPS)
        log_info("TorchDevice (apply_patches ops.device.cuda): PyTorch compiled with CUDA, applying redirection patches for non-CUDA hardware.")
        # Apply specific redirection versions of functions. Many of these are already covered by the _replacement functions.
        # This section ensures that if PyTorch *has* the CUDA functions, they are appropriately no-oped or redirected.
        # Some stubs from the 'non_cuda_build' list might be useful here too if the original is problematic on MPS.

        _cuda_module_ref.device_count = lambda: 1 if hardware_info.is_native_mps_available() else (t_cuda_device_count() if t_cuda_device_count else 0)
        _cuda_module_ref.current_device = t_cuda_current_device if t_cuda_current_device else (lambda: 0)
        _cuda_module_ref.get_device_name = t_cuda_get_device_name if t_cuda_get_device_name else (lambda device=None: "MPS (Redirected by TorchDevice)" if hardware_info.is_native_mps_available() else "CUDA (Simulated by TorchDevice)")
        _cuda_module_ref.get_device_capability = t_cuda_get_device_capability if t_cuda_get_device_capability else (lambda device=None: (1,0) if hardware_info.is_native_mps_available() else (7,0) )
        _cuda_module_ref.get_device_properties = cuda_get_device_properties_replacement # Always use our simulation for consistency on non-CUDA
        _cuda_module_ref.set_device = cuda_set_device_replacement # No-op or redirects
        
        _cuda_module_ref.empty_cache = cuda_empty_cache_replacement
        _cuda_module_ref.memory_allocated = cuda_memory_allocated_replacement
        _cuda_module_ref.max_memory_allocated = cuda_max_memory_allocated_replacement
        _cuda_module_ref.reset_peak_memory_stats = cuda_reset_peak_memory_stats_replacement
        _cuda_module_ref.memory_reserved = cuda_memory_reserved_replacement
        _cuda_module_ref.max_memory_reserved = cuda_max_memory_reserved_replacement
        _cuda_module_ref.memory_stats = cuda_memory_stats_replacement
        _cuda_module_ref.memory_snapshot = getattr(_cuda_module_ref, 'memory_snapshot', lambda: []) # Use original if exists, else stub

        _cuda_module_ref.get_arch_list = getattr(_cuda_module_ref, 'get_arch_list', lambda: ['sm_70'] if hardware_info.is_mps_available() else [])
        _cuda_module_ref.get_gencode_flags = getattr(_cuda_module_ref, 'get_gencode_flags', lambda: "")


        # AMP - ensure replacements are used if running on MPS/CPU
        if hasattr(_cuda_module_ref, 'amp') and isinstance(getattr(_cuda_module_ref, 'amp'), types.ModuleType):
            _cuda_module_ref.amp.autocast = _cuda_amp_autocast_replacement
            _cuda_module_ref.amp.GradScaler = _GradScalerReplacement
        elif not hasattr(_cuda_module_ref, 'amp'): # Create amp if it's missing even in CUDA-compiled PyTorch
            _cuda_module_ref.amp = types.ModuleType('torch.cuda.amp')
            _cuda_module_ref.amp.autocast = _cuda_amp_autocast_replacement
            _cuda_module_ref.amp.GradScaler = _GradScalerReplacement
            log_warning("TorchDevice (apply_patches ops.device.cuda): torch.cuda.amp submodule was missing in CUDA-compiled PyTorch; created and stubbed for MPS/CPU.")
        
        # Ensure other stubs from test_cuda_stubs are present if not already handled
        # This is a safety net for items specifically in test_cuda_stubs that might not be covered by general replacements
        # when PyTorch is CUDA-compiled but we are on MPS/CPU.
        # Example: 'list_gpu_processes', 'utilization', etc.
        # We can reuse parts of stubs_for_non_cuda_build dictionary if needed.
        # For now, the above covers the main redirection cases.
        # Specific stubs from test_cuda_stubs for this scenario:
        _cuda_module_ref.list_gpu_processes = getattr(_cuda_module_ref, 'list_gpu_processes', lambda *a, **kw: [])
        _cuda_module_ref.utilization = getattr(_cuda_module_ref, 'utilization', lambda *a, **kw: 0)
        _cuda_module_ref.mem_get_info = getattr(_cuda_module_ref, 'mem_get_info', lambda *a, **kw: (0,0))
        _cuda_module_ref.reset_accumulated_memory_stats = getattr(_cuda_module_ref, 'reset_accumulated_memory_stats', lambda *a, **kw: None)
        _cuda_module_ref.reset_max_memory_cached = getattr(_cuda_module_ref, 'reset_max_memory_cached', lambda *a, **kw: None)
        _cuda_module_ref.caching_allocator_alloc = getattr(_cuda_module_ref, 'caching_allocator_alloc', lambda *a, **kw: 0)
        _cuda_module_ref.caching_allocator_delete = getattr(_cuda_module_ref, 'caching_allocator_delete', lambda *a, **kw: None)
        _cuda_module_ref.get_allocator_backend = getattr(_cuda_module_ref, 'get_allocator_backend', lambda *a, **kw: "mps_redirect_backend")
        _cuda_module_ref.change_current_allocator = getattr(_cuda_module_ref, 'change_current_allocator', lambda *a, **kw: None)
        _cuda_module_ref.make_graphed_callables = getattr(_cuda_module_ref, 'make_graphed_callables', lambda *a, **kw: [])
        _cuda_module_ref.is_current_stream_capturing = getattr(_cuda_module_ref, 'is_current_stream_capturing', lambda *a, **kw: False)
        _cuda_module_ref.graph_pool_handle = getattr(_cuda_module_ref, 'graph_pool_handle', lambda *a, **kw: None)
        _cuda_module_ref.can_device_access_peer = getattr(_cuda_module_ref, 'can_device_access_peer', lambda *a, **kw: False)
        _cuda_module_ref.current_blas_handle = getattr(_cuda_module_ref, 'current_blas_handle', lambda *a, **kw: None)
        _cuda_module_ref.memory_usage = getattr(_cuda_module_ref, 'memory_usage', lambda *a, **kw: 0)
        _cuda_module_ref.temperature = getattr(_cuda_module_ref, 'temperature', lambda *a, **kw: 0)
        _cuda_module_ref.power_draw = getattr(_cuda_module_ref, 'power_draw', lambda *a, **kw: 0)
        _cuda_module_ref.clock_rate = getattr(_cuda_module_ref, 'clock_rate', lambda *a, **kw: 0)
        _cuda_module_ref.set_sync_debug_mode = getattr(_cuda_module_ref, 'set_sync_debug_mode', lambda *a, **kw: None)
        _cuda_module_ref.get_sync_debug_mode = getattr(_cuda_module_ref, 'get_sync_debug_mode', lambda *a, **kw: 0)

        # Ensure Stream and Event are minimally stubbed if not fully patched by their dedicated modules
        # This is a fallback for the CUDA-compiled but non-CUDA hardware case.
        if not hasattr(getattr(_cuda_module_ref, 'Stream', object), '_torchdevice_patched'):
            _cuda_module_ref.Stream = type('Stream', (object,), {'__init__': lambda s, *a, **k: None, 'synchronize': lambda s: None, 'query': lambda s: True, 'record_event': lambda s, e=None: e if e else type('Event', (object,), {'record':lambda ev:None})(), 'wait_event': lambda s,e: None, 'wait_stream': lambda s,st: None, '__enter__': lambda s: s, '__exit__': lambda s,*a: None })
            log_info("TorchDevice (apply_patches ops.device.cuda): Applied minimal Stream stub for CUDA-compiled/non-CUDA hardware.")
        if not hasattr(getattr(_cuda_module_ref, 'Event', object), '_torchdevice_patched'):
            _cuda_module_ref.Event = type('Event', (object,), {'__init__': lambda s, *a, **k: None, 'record': lambda s, st=None: None, 'synchronize': lambda s: None, 'query': lambda s: True, 'wait': lambda s, st=None: None, 'elapsed_time': lambda s,e: 0.0})
            log_info("TorchDevice (apply_patches ops.device.cuda): Applied minimal Event stub for CUDA-compiled/non-CUDA hardware.")
        _cuda_module_ref.current_stream = getattr(_cuda_module_ref, 'current_stream', lambda *a, **kw: None)
        _cuda_module_ref.default_stream = getattr(_cuda_module_ref, 'default_stream', lambda *a, **kw: None)

        log_info("TorchDevice (apply_patches ops.device.cuda): Applied redirection patches for CUDA-compiled PyTorch on non-CUDA hardware.")

    log_info("TorchDevice (apply_patches ops.device.cuda): CUDA patches application finished.")

# --- Missing AMP replacement functions ---
def _cuda_amp_autocast_replacement(enabled=True, dtype=None, cache_enabled=None, **kwargs):
    """
    Replacement for torch.cuda.amp.autocast that works on MPS.
    
    This creates a context manager for autocasting in PyTorch's automatic mixed precision.
    When running on MPS, this redirects to torch.amp.autocast(device_type='mps').
    """
    from ...core.device import DeviceManager # Local import to break circular dependency
    log_info("TorchDevice (_cuda_amp_autocast_replacement): Redirecting cuda.amp.autocast to amp.autocast")
    device = DeviceManager.get_default_device()
    device_type = 'cuda' if device and device.type == 'cuda' else 'mps' if device and device.type == 'mps' else 'cpu'
    
    # For MPS, use the general torch.amp.autocast with device_type='mps'
    if device_type == 'mps':
        log_info("TorchDevice (_cuda_amp_autocast_replacement): Using MPS autocast")
        import torch.amp
        return torch.amp.autocast(device_type='mps', dtype=dtype, cache_enabled=cache_enabled, **kwargs)
    elif device_type == 'cpu':
        log_info("TorchDevice (_cuda_amp_autocast_replacement): Using CPU autocast")
        import torch.amp
        return torch.amp.autocast(device_type='cpu', dtype=dtype, cache_enabled=cache_enabled, **kwargs)
    else:  # device_type == 'cuda'
        log_info("TorchDevice (_cuda_amp_autocast_replacement): Using original CUDA autocast")
        # Use the original function for CUDA
        if t_cuda_amp_autocast:
            return t_cuda_amp_autocast(enabled=enabled, dtype=dtype, cache_enabled=cache_enabled, **kwargs)
        else:
            # Fallback if original not available
            import torch.amp
            return torch.amp.autocast(device_type='cuda', dtype=dtype, cache_enabled=cache_enabled, **kwargs)


class _GradScalerReplacement:
    """
    Replacement for torch.cuda.amp.GradScaler that works on MPS.
    
    This creates a GradScaler compatible with MPS devices by redirecting to torch.amp.GradScaler
    with appropriate arguments.  
    """
    def __init__(self, **kwargs):
        from ...core.device import DeviceManager # Local import to break circular dependency
        log_info("TorchDevice (_GradScalerReplacement.__init__): Creating compatible GradScaler")
        device = DeviceManager.get_default_device()
        device_type = device.type if device else 'cpu'
        
        # Use the standard torch.amp.GradScaler
        import torch.amp
        self._grad_scaler = torch.amp.GradScaler(**kwargs)
    
    def __getattr__(self, name):
        # Pass through all attribute access to the underlying GradScaler
        return getattr(self._grad_scaler, name)


# Memory management functions are now in memory.py
# Use the consolidated implementations from memory module
# instead of duplicating the functionality here

# Module initialization
log_info("Initializing TorchDevice CUDA operations module")

__all__: list[str] = [
    'apply_patches'
]

log_info("TorchDevice CUDA operations module initialized") 