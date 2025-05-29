"""
TorchDevice Memory Emulation Module
---------------------------------
Memory emulation and statistics for non-CUDA devices.
"""

import psutil
import os
from typing import Optional, Tuple, Dict, Any
from ...core.logger import auto_log

# --- Internal Memory-related Emulation Functions ---

@auto_log()
def _memory_allocated(device: Optional[int] = None) -> int:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss

@auto_log()
def _memory_reserved(device: Optional[int] = None) -> int:
    return psutil.virtual_memory().total

@auto_log()
def _max_memory_allocated(device: Optional[int] = None) -> int:
    return _memory_allocated(device)

@auto_log()
def _max_memory_reserved(device: Optional[int] = None) -> int:
    return _memory_reserved(device)

@auto_log()
def _mem_get_info(device: Optional[int] = None) -> Tuple[int, int]:
    vm = psutil.virtual_memory()
    return vm.available, vm.total

@auto_log()
def _memory_stats(device: Optional[int] = None) -> Dict[str, Any]:
    allocated = _memory_allocated(device)
    reserved = _memory_reserved(device)
    vm = psutil.virtual_memory()
    
    # Match the format expected by torch.cuda.memory_stats()
    stats = {
        'allocated_bytes.all.current': allocated,
        'allocated_bytes.all.peak': allocated,
        'allocated_bytes.all.allocated': allocated,
        'allocated_bytes.all.freed': 0,
        'active_bytes.all.current': allocated,
        'active_bytes.all.peak': allocated,
        'active_bytes.all.allocated': allocated,
        'active_bytes.all.freed': 0,
        'requested_bytes.all.current': allocated,
        'requested_bytes.all.peak': allocated,
        'requested_bytes.all.allocated': allocated,
        'requested_bytes.all.freed': 0,
        'reserved_bytes.all.current': reserved,
        'reserved_bytes.all.peak': reserved,
        'reserved_bytes.all.allocated': reserved,
        'reserved_bytes.all.freed': 0,
        'inactive_split_bytes.all.current': 0,
        'inactive_split_bytes.all.peak': 0,
        'inactive_split_bytes.all.allocated': 0,
        'inactive_split_bytes.all.freed': 0,
        'allocation.all.current': 1,
        'allocation.all.peak': 1,
        'allocation.all.allocated': 1,
        'allocation.all.freed': 0,
        'active.all.current': 1,
        'active.all.peak': 1,
        'active.all.allocated': 1,
        'active.all.freed': 0,
        'segment.all.current': 1,
        'segment.all.peak': 1,
        'segment.all.allocated': 1,
        'segment.all.freed': 0,
        'inactive_split.all.current': 0,
        'inactive_split.all.peak': 0,
        'inactive_split.all.allocated': 0,
        'inactive_split.all.freed': 0,
        'num_alloc_retries': 0,
        'num_ooms': 0,
    }
    return stats

@auto_log()
def _memory_snapshot(device: Optional[int] = None):
    return [{
        'device': 0,
        'address': 0,
        'total_size': _memory_allocated(device),
        'allocated_size': _memory_allocated(device),
        'active': True,
        'segment_type': 'small_pool',
    }]

@auto_log()
def _memory_summary(device: Optional[int] = None, abbreviated: bool = False) -> str:
    stats = _memory_stats(device)
    return (f"Memory Allocated: {stats['allocated_bytes.all.current']} bytes\n"
            f"Memory Reserved: {stats['reserved_bytes.all.current']} bytes\n"
            f"Memory Free: {psutil.virtual_memory().available} bytes\n"
            f"Memory Total: {psutil.virtual_memory().total} bytes\n")

@auto_log()
def _empty_cache(device: Optional[int] = None) -> None:
    pass

@auto_log()
def _reset_peak_memory_stats(device: Optional[int] = None) -> None:
    pass

@auto_log()
def _reset_accumulated_memory_stats(device: Optional[int] = None) -> None:
    pass

@auto_log()
def _reset_max_memory_allocated(device: Optional[int] = None) -> None:
    pass

@auto_log()
def _reset_max_memory_reserved(device: Optional[int] = None) -> None:
    pass 