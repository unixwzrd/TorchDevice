"""
TorchDevice CUDA Unassigned/Stub Functions
-----------------------------------------
This module centralizes all unsupported or stubbed CUDA functions for non-CUDA backends (e.g., MPS, CPU).
All functions listed in CUDA_STUBS are patched as no-ops on torch.cuda.

Naming Convention:
- Internal helper functions use the same name as the public API but with a leading underscore (e.g., _set_stream for torch.cuda.set_stream).
- Public API functions (patched onto torch.cuda) do not use the underscore and are thin wrappers around the internal helpers.
"""

from typing import Callable, Dict


def make_noop(name: str) -> Callable:
    """Return a no-op function for the given CUDA function name."""
    def _noop(*args, **kwargs):
        """Stub for torch.cuda.{0}(). No-op on non-CUDA backends.""".format(name)
        return None
    _noop.__name__ = name
    return _noop


# List of unsupported CUDA function names to stub (expanded from TorchDevice.py)
CUDA_STUBS = [
    'set_stream', 'mem_get_info', 'reset_accumulated_memory_stats', 'reset_max_memory_allocated',
    'reset_max_memory_cached', 'caching_allocator_alloc', 'caching_allocator_delete', 'get_allocator_backend',
    'change_current_allocator', 'nvtx', 'jiterator', 'graph', 'CUDAGraph', 'make_graphed_callables',
    'is_current_stream_capturing', 'graph_pool_handle', 'can_device_access_peer', 'comm', 'get_gencode_flags',
    'current_blas_handle', 'memory_usage', 'utilization', 'temperature', 'power_draw', 'clock_rate',
    'set_sync_debug_mode', 'get_sync_debug_mode', 'list_gpu_processes', 'seed', 'seed_all', 'manual_seed',
    'manual_seed_all', 'get_rng_state', 'get_rng_state_all', 'set_rng_state', 'set_rng_state_all', 'initial_seed',
    'ipc_collect'
]

# Create a dictionary of stub functions
cuda_stub_functions: Dict[str, Callable] = {name: make_noop(name) for name in CUDA_STUBS}

def apply_patches() -> None:
    import torch
    for name, fn in cuda_stub_functions.items():
        setattr(torch.cuda, name, fn)
    # Patch additional no-ops for unsupported CUDA features with plausible return types
    torch.cuda.utilization = lambda *a, **kw: 0
    torch.cuda.temperature = lambda *a, **kw: 0
    torch.cuda.power_draw = lambda *a, **kw: 0
    torch.cuda.clock_rate = lambda *a, **kw: 0
    torch.cuda.set_sync_debug_mode = lambda *a, **kw: None
    torch.cuda.get_sync_debug_mode = lambda *a, **kw: 0
    torch.cuda.list_gpu_processes = lambda *a, **kw: [] 