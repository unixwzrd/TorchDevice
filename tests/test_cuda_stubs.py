import torch
import pytest

# (function_name, expected_type_or_value)
CUDA_FUNCTIONS = [
    ('set_stream', None),
    ('mem_get_info', tuple),
    ('reset_accumulated_memory_stats', None),
    ('reset_max_memory_allocated', None),
    ('reset_max_memory_cached', None),
    ('caching_allocator_alloc', None),
    ('caching_allocator_delete', None),
    ('get_allocator_backend', None),
    ('change_current_allocator', None),
    ('nvtx', None),
    ('jiterator', None),
    ('graph', None),
    ('CUDAGraph', None),
    ('make_graphed_callables', None),
    ('is_current_stream_capturing', None),
    ('graph_pool_handle', None),
    ('can_device_access_peer', None),
    ('comm', None),
    ('get_gencode_flags', None),
    ('current_blas_handle', None),
    ('memory_usage', None),
    ('utilization', None),
    ('temperature', None),
    ('power_draw', None),
    ('clock_rate', None),
    ('set_sync_debug_mode', None),
    ('get_sync_debug_mode', None),
    ('list_gpu_processes', None),
    ('seed', None),
    ('seed_all', None),
    ('manual_seed', None),
    ('manual_seed_all', None),
    ('get_rng_state', None),
    ('get_rng_state_all', None),
    ('set_rng_state', None),
    ('set_rng_state_all', None),
    ('initial_seed', None),
]

@pytest.mark.parametrize("func_name,expected", CUDA_FUNCTIONS)
def test_cuda_function(func_name, expected):
    fn = getattr(torch.cuda, func_name, None)
    assert fn is not None, f"torch.cuda.{func_name} is missing"
    result = fn()
    if expected is None:
        assert result is None, f"torch.cuda.{func_name} should return None"
    elif isinstance(expected, type):
        assert isinstance(result, expected), f"torch.cuda.{func_name} should return {expected}"
    else:
        assert result == expected, f"torch.cuda.{func_name} should return {expected}" 