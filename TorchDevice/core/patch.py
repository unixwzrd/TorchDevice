"""
TorchDevice Core Patch Module
------------------------
Core patching functionality for TorchDevice.
"""

import torch
from typing import List, Any
from .logger import log_info, auto_log
from .device import apply_patches as apply_device_patches

log_info("Importing TorchDevice/core/patch.py")

# Track patch status
_patches_applied = False

def _mock_cuda_init():
    """Mock CUDA initialization to prevent _lazy_init errors."""
    pass

def _mock_cuda_synchronize():
    """Mock CUDA synchronize function."""
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize()

def _mock_cuda_current_device() -> int:
    """Mock current CUDA device function."""
    return 0

def _mock_cuda_device_count() -> int:
    """Mock CUDA device count function."""
    return 1

def _mock_cuda_is_available() -> bool:
    """Mock CUDA availability function."""
    return True

def _mock_cuda_get_device_properties(device: Any) -> Any:
    """Mock CUDA device properties."""
    class MockProperties:
        def __init__(self):
            self.name = "Mock GPU"
            self.total_memory = 1024 * 1024 * 1024  # 1GB
            self.major = 1
            self.minor = 0
    return MockProperties()

@auto_log()
def apply_all_patches() -> None:
    """Apply all patches to PyTorch functions."""
    global _patches_applied
    
    if _patches_applied:
        return
        
    log_info("[TorchDevice] patch.apply_all_patches called")

    # Apply device patches first
    apply_device_patches()

    # Apply device context patches
    log_info("Applying device context patches")
    from ..ops.device import apply_patches as apply_device_context_patches
    apply_device_context_patches()

    # Apply memory patches
    log_info("Applying memory patches")
    from ..ops.memory import apply_patches as apply_memory_patches
    apply_memory_patches()

    # Apply random number generation patches
    log_info("Applying random number generation patches")
    from ..ops.random import apply_patches as apply_random_patches
    apply_random_patches()

    # Apply stream patches
    log_info("Applying stream patches")
    from ..ops.streams.cuda import apply_patches as apply_cuda_stream_patches
    from ..ops.streams.mps import apply_patches as apply_mps_stream_patches
    from ..ops.streams.synchronize import apply_patches as apply_stream_sync_patches
    apply_cuda_stream_patches()
    apply_mps_stream_patches()
    apply_stream_sync_patches()

    # Apply event patches
    log_info("Applying event patches")
    from ..ops.events.cuda_events import apply_patches as apply_cuda_event_patches
    from ..ops.events.mps_events import apply_patches as apply_mps_event_patches
    from ..ops.events.synchronize import apply_patches as apply_event_sync_patches
    apply_cuda_event_patches()
    apply_mps_event_patches()
    apply_event_sync_patches()

    # Apply neural network patches
    log_info("Applying neural network patches")
    from ..ops.nn import apply_patches as apply_nn_patches
    apply_nn_patches()

    # Apply autograd patches
    log_info("Applying autograd patches")
    from ..ops.autograd.function import apply_patches as apply_autograd_function_patches
    from ..ops.autograd.variable import apply_patches as apply_autograd_variable_patches
    from ..ops.autograd.grad_mode import apply_patches as apply_grad_mode_patches
    apply_autograd_function_patches()
    apply_autograd_variable_patches()
    apply_grad_mode_patches()

    # Apply optimization patches
    log_info("Applying optimization patches")
    from ..ops.optim.optimizer import apply_patches as apply_optimizer_patches
    from ..ops.optim.lr_scheduler import apply_patches as apply_lr_scheduler_patches
    apply_optimizer_patches()
    apply_lr_scheduler_patches()

    # Apply tensor patches
    log_info("Applying tensor patches")
    from ..ops.tensor.creation import apply_patches as apply_tensor_creation_patches
    apply_tensor_creation_patches()

    # Apply CUDA stubs for unsupported functions
    def make_noop(name: str):
        def _noop(*args, **kwargs):
            return None
        _noop.__name__ = name
        return _noop

    # Patch core CUDA functionality
    torch.cuda._lazy_init = _mock_cuda_init
    torch.cuda.synchronize = _mock_cuda_synchronize
    torch.cuda.current_device = _mock_cuda_current_device
    torch.cuda.device_count = _mock_cuda_device_count
    torch.cuda.is_available = _mock_cuda_is_available
    torch.cuda.get_device_properties = _mock_cuda_get_device_properties

    cuda_stubs = [
        'set_stream', 'mem_get_info', 'reset_accumulated_memory_stats', 'reset_max_memory_allocated',
        'reset_max_memory_cached', 'caching_allocator_alloc', 'caching_allocator_delete', 'get_allocator_backend',
        'change_current_allocator', 'nvtx', 'jiterator', 'graph', 'CUDAGraph', 'make_graphed_callables',
        'is_current_stream_capturing', 'graph_pool_handle', 'can_device_access_peer', 'comm', 'get_gencode_flags',
        'current_blas_handle', 'memory_usage', 'utilization', 'temperature', 'power_draw', 'clock_rate',
        'set_sync_debug_mode', 'get_sync_debug_mode', 'list_gpu_processes', 'seed', 'seed_all', 'manual_seed',
        'manual_seed_all', 'get_rng_state', 'get_rng_state_all', 'set_rng_state', 'set_rng_state_all', 'initial_seed',
        'ipc_collect'
    ]

    for name in cuda_stubs:
        if not hasattr(torch.cuda, name):
            setattr(torch.cuda, name, make_noop(name))

    # Add plausible return values for some functions
    torch.cuda.utilization = lambda *a, **kw: 0
    torch.cuda.temperature = lambda *a, **kw: 0
    torch.cuda.power_draw = lambda *a, **kw: 0
    torch.cuda.clock_rate = lambda *a, **kw: 0
    torch.cuda.set_sync_debug_mode = lambda *a, **kw: None
    torch.cuda.get_sync_debug_mode = lambda *a, **kw: 0
    torch.cuda.list_gpu_processes = lambda *a, **kw: []

    _patches_applied = True
    log_info("All patches applied successfully")


@auto_log()
def ensure_patches_applied() -> None:
    """Ensure all patches are applied."""
    if not _patches_applied:
        apply_all_patches()


__all__: List[str] = [
    'apply_all_patches',
    'ensure_patches_applied'
]
