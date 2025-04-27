"""
TorchDevice Memory Management and Patching
-----------------------------------------
All memory management, emulation, and patching logic for TorchDevice is centralized in this module.


"""

import psutil
import os
from typing import Optional, Tuple, Dict, Any
from ..modules.TDLogger import auto_log

# --- Internal Memory-related Emulation Functions (use leading underscore) ---

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
    stats = {
        'active.all.current': allocated,
        'active.all.peak': _max_memory_allocated(device),
        'reserved_bytes.all.current': reserved,
        'reserved_bytes.all.peak': _max_memory_reserved(device),
        'allocated': allocated,
        'reserved': reserved,
        'free': psutil.virtual_memory().available,
        'total': psutil.virtual_memory().total,
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
    return (f"Memory Allocated: {stats['allocated']} bytes\n"
            f"Memory Reserved: {stats['reserved']} bytes\n"
            f"Memory Free: {stats['free']} bytes\n"
            f"Memory Total: {stats['total']} bytes\n")

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

@auto_log()
def _empty_cache(device: Optional[int] = None):
    import torch
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

@auto_log()
def mock_cuda_reset_peak_memory_stats(cls, device=None):
    pass

@auto_log()
def mock_cuda_reset_accumulated_memory_stats(cls, device=None):
    pass

@auto_log()
def mock_cuda_reset_max_memory_allocated(cls, device=None):
    pass

@auto_log()
def mock_cuda_reset_max_memory_reserved(cls, device=None):
    pass

# --- Public API functions (patched onto torch.cuda) ---
def memory_allocated(device: Optional[int] = None) -> int:
    return _memory_allocated(device)
def memory_reserved(device: Optional[int] = None) -> int:
    return _memory_reserved(device)
def max_memory_allocated(device: Optional[int] = None) -> int:
    return _max_memory_allocated(device)
def max_memory_reserved(device: Optional[int] = None) -> int:
    return _max_memory_reserved(device)
def mem_get_info(device: Optional[int] = None) -> Tuple[int, int]:
    return _mem_get_info(device)
def memory_stats(device: Optional[int] = None) -> Dict[str, Any]:
    return _memory_stats(device)
def memory_snapshot(device: Optional[int] = None):
    return _memory_snapshot(device)
def memory_summary(device: Optional[int] = None, abbreviated: bool = False) -> str:
    return _memory_summary(device, abbreviated)
def reset_peak_memory_stats(device: Optional[int] = None) -> None:
    return _reset_peak_memory_stats(device)
def reset_accumulated_memory_stats(device: Optional[int] = None) -> None:
    return _reset_accumulated_memory_stats(device)
def reset_max_memory_allocated(device: Optional[int] = None) -> None:
    return _reset_max_memory_allocated(device)
def reset_max_memory_reserved(device: Optional[int] = None) -> None:
    return _reset_max_memory_reserved(device)
def empty_cache(device: Optional[int] = None):
    return _empty_cache(device)

def apply_patches() -> None:
    import torch
    torch.cuda.memory_allocated = memory_allocated
    torch.cuda.memory_reserved = memory_reserved
    torch.cuda.max_memory_allocated = max_memory_allocated
    torch.cuda.max_memory_reserved = max_memory_reserved
    torch.cuda.mem_get_info = mem_get_info
    torch.cuda.memory_stats = memory_stats
    torch.cuda.memory_summary = memory_summary
    torch.cuda.memory_snapshot = memory_snapshot
    torch.cuda.empty_cache = empty_cache
    torch.cuda.reset_peak_memory_stats = reset_peak_memory_stats
    torch.cuda.reset_accumulated_memory_stats = reset_accumulated_memory_stats
    torch.cuda.reset_max_memory_allocated = reset_max_memory_allocated
    torch.cuda.reset_max_memory_reserved = reset_max_memory_reserved 