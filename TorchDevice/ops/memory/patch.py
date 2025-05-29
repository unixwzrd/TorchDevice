"""
TorchDevice Memory Patch Module
-----------------------------
Patches PyTorch's CUDA memory functions with emulated versions.
"""

import torch
from typing import Optional, Dict, Any, Tuple
from .emulation import (
    _memory_allocated, _memory_reserved, _max_memory_allocated, _max_memory_reserved,
    _mem_get_info, _memory_stats, _memory_summary, _memory_snapshot, _empty_cache,
    _reset_peak_memory_stats, _reset_accumulated_memory_stats,
    _reset_max_memory_allocated, _reset_max_memory_reserved
)
from ...core.logger import auto_log

@auto_log()
def memory_allocated(device: Optional[int] = None) -> int:
    return _memory_allocated(device)

@auto_log()
def memory_reserved(device: Optional[int] = None) -> int:
    return _memory_reserved(device)

@auto_log()
def max_memory_allocated(device: Optional[int] = None) -> int:
    return _max_memory_allocated(device)

@auto_log()
def max_memory_reserved(device: Optional[int] = None) -> int:
    return _max_memory_reserved(device)

@auto_log()
def mem_get_info(device: Optional[int] = None) -> Tuple[int, int]:
    return _mem_get_info(device)

@auto_log()
def memory_stats(device: Optional[int] = None) -> Dict[str, Any]:
    return _memory_stats(device)

@auto_log()
def memory_summary(device: Optional[int] = None, abbreviated: bool = False) -> str:
    return _memory_summary(device, abbreviated)

@auto_log()
def memory_snapshot(device: Optional[int] = None):
    return _memory_snapshot(device)

@auto_log()
def empty_cache(device: Optional[int] = None) -> None:
    return _empty_cache(device)

@auto_log()
def reset_peak_memory_stats(device: Optional[int] = None) -> None:
    return _reset_peak_memory_stats(device)

@auto_log()
def reset_accumulated_memory_stats(device: Optional[int] = None) -> None:
    return _reset_accumulated_memory_stats(device)

@auto_log()
def reset_max_memory_allocated(device: Optional[int] = None) -> None:
    return _reset_max_memory_allocated(device)

@auto_log()
def reset_max_memory_reserved(device: Optional[int] = None) -> None:
    return _reset_max_memory_reserved(device)

def apply_patches() -> None:
    """Apply all memory-related patches to torch.cuda."""
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