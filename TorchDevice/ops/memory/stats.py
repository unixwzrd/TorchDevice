"""
TorchDevice Memory Stats Module
---------------------------
Memory usage tracking and statistics.
"""

import torch
from typing import Optional
from TorchDevice.core.logger import log_info, auto_log

# Store original functions
t_cuda_memory_allocated = torch.cuda.memory_allocated if hasattr(torch.cuda, 'memory_allocated') else None
t_cuda_memory_reserved = torch.cuda.memory_reserved if hasattr(torch.cuda, 'memory_reserved') else None
t_cuda_max_memory_allocated = torch.cuda.max_memory_allocated if hasattr(torch.cuda, 'max_memory_allocated') else None
t_cuda_max_memory_reserved = torch.cuda.max_memory_reserved if hasattr(torch.cuda, 'max_memory_reserved') else None

# MPS memory functions
t_mps_current_allocated_memory = torch.mps.current_allocated_memory if hasattr(torch.mps, 'current_allocated_memory') else None
t_mps_driver_allocated_memory = torch.mps.driver_allocated_memory if hasattr(torch.mps, 'driver_allocated_memory') else None

# Original CUDA reset functions
t_cuda_reset_peak_memory_stats = torch.cuda.reset_peak_memory_stats if hasattr(torch.cuda, 'reset_peak_memory_stats') else None
t_cuda_reset_accumulated_memory_stats = torch.cuda.reset_accumulated_memory_stats if hasattr(torch.cuda, 'reset_accumulated_memory_stats') else None
t_cuda_mem_get_info = torch.cuda.mem_get_info if hasattr(torch.cuda, 'mem_get_info') else None
t_cuda_memory_stats = torch.cuda.memory_stats if hasattr(torch.cuda, 'memory_stats') else None
t_cuda_memory_summary = torch.cuda.memory_summary if hasattr(torch.cuda, 'memory_summary') else None


@auto_log()
def memory_allocated(device: Optional[torch.device] = None) -> int:
    """Return the current GPU memory occupied by tensors in bytes."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_memory_allocated:
        return t_cuda_memory_allocated(device)
    elif device.type == 'mps' and t_mps_current_allocated_memory:
        return t_mps_current_allocated_memory()
    return 0


@auto_log()
def memory_reserved(device: Optional[torch.device] = None) -> int:
    """Return the current GPU memory managed by the caching allocator in bytes."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_memory_reserved:
        return t_cuda_memory_reserved(device)
    elif device.type == 'mps' and t_mps_driver_allocated_memory:
        return t_mps_driver_allocated_memory()
    return 0


@auto_log()
def max_memory_allocated(device: Optional[torch.device] = None) -> int:
    """Return the maximum GPU memory occupied by tensors in bytes."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_max_memory_allocated:
        return t_cuda_max_memory_allocated(device)
    elif device.type == 'mps' and t_mps_current_allocated_memory:  # Use current as max for MPS
        return t_mps_current_allocated_memory()
    return 0


@auto_log()
def max_memory_reserved(device: Optional[torch.device] = None) -> int:
    """Return the maximum GPU memory managed by the caching allocator in bytes."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_max_memory_reserved:
        return t_cuda_max_memory_reserved(device)
    elif device.type == 'mps' and t_mps_driver_allocated_memory:  # Use current driver allocated as max for MPS
        return t_mps_driver_allocated_memory()
    return 0


@auto_log()
def reset_peak_memory_stats(device: Optional[torch.device] = None) -> None:
    """Reset the peak memory stats for a given device. No-op for MPS."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_reset_peak_memory_stats:
        t_cuda_reset_peak_memory_stats(device)
    elif device.type == 'mps':
        log_info("MPS does not support resetting peak memory stats. This call is a no-op.")
    # No action for other device types or if original function is not available

@auto_log()
def reset_accumulated_memory_stats(device: Optional[torch.device] = None) -> None:
    """Reset the accumulated memory stats for a given device. No-op for MPS."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_reset_accumulated_memory_stats:
        t_cuda_reset_accumulated_memory_stats(device)
    elif device.type == 'mps':
        log_info("MPS does not support resetting accumulated memory stats. This call is a no-op.")
    # No action for other device types or if original function is not available

@auto_log()
def mem_get_info(device: Optional[torch.device] = None) -> tuple[int, int]:
    """Return the free and total GPU memory. (free, total)"""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_mem_get_info:
        return t_cuda_mem_get_info(device)
    elif device.type == 'mps':
        total_memory = t_mps_driver_allocated_memory() if t_mps_driver_allocated_memory else 0
        current_allocated = t_mps_current_allocated_memory() if t_mps_current_allocated_memory else 0
        free_memory = total_memory - current_allocated
        return free_memory, total_memory
    return 0, 0

@auto_log()
def memory_stats(device: Optional[torch.device] = None) -> dict[str, int]:
    """Return a dictionary of memory statistics."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_memory_stats:
        return t_cuda_memory_stats(device)
    elif device.type == 'mps':
        # Emulate common keys from torch.cuda.memory_stats()
        # Note: MPS does not provide the same level of detail as CUDA's allocator.
        current_alloc = memory_allocated(device) # Uses t_mps_current_allocated_memory
        current_reserved = memory_reserved(device) # Uses t_mps_driver_allocated_memory
        peak_alloc = max_memory_allocated(device) # Uses t_mps_current_allocated_memory as peak
        peak_reserved = max_memory_reserved(device) # Uses t_mps_driver_allocated_memory as peak

        # Basic stats matching some common CUDA keys
        # CUDA keys often have '.all.current', '.all.peak', etc. suffixes
        # We'll provide a simplified version for MPS
        stats_dict = {
            "allocated_bytes.all.current": current_alloc,
            "reserved_bytes.all.current": current_reserved,
            "allocated_bytes.all.peak": peak_alloc,
            "reserved_bytes.all.peak": peak_reserved,
            # CUDA has many more detailed stats like 'active_bytes', 'inactive_split_bytes', etc.
            # These are not directly available for MPS, so we'll omit them or set to 0 if essential.
            # For now, focusing on the primary ones.
            "num_alloc_retries": 0, # Placeholder
            "num_ooms": 0, # Placeholder
            # Add other relevant stats if they can be derived or are meaningful for MPS
        }
        # Add any other simple stats that can be derived
        stats_dict["active_bytes.all.current"] = current_alloc # Simplification: active is current allocated
        stats_dict["active_bytes.all.peak"] = peak_alloc # Simplification
        stats_dict["inactive_split_bytes.all.current"] = current_reserved - current_alloc # Simplification
        stats_dict["inactive_split_bytes.all.peak"] = peak_reserved - peak_alloc # Simplification

        return stats_dict
    return {}

@auto_log()
def memory_summary(device: Optional[torch.device] = None, abbreviated: bool = False) -> str:
    """Return a human-readable memory summary."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = device or DeviceManager.get_default_device()
    if device.type == 'cuda' and t_cuda_memory_summary:
        return t_cuda_memory_summary(device, abbreviated=abbreviated)
    elif device.type == 'mps':
        stats = memory_stats(device)
        if not stats:
            return "MPS: No memory statistics available."

        # Simplified summary similar to CUDA's output
        summary_lines = [
            f"MPS Memory Summary (Device: {device.index if device.index is not None else 'default'})",
            "-----------------------------------------------------------------------",
            f"| {'Metric':<35} | {'Value':>25} |",
            "|-----------------------------------------|---------------------------|"
        ]

        def format_value(key):
            val_bytes = stats.get(key, 0)
            if val_bytes > 1024 * 1024 * 1024:
                return f"{val_bytes / (1024**3):.2f} GiB"
            elif val_bytes > 1024 * 1024:
                return f"{val_bytes / (1024**2):.2f} MiB"
            elif val_bytes > 1024:
                return f"{val_bytes / 1024:.2f} KiB"
            return f"{val_bytes} B"

        summary_lines.append(f"| {'Allocated memory current':<35} | {format_value('allocated_bytes.all.current'):>25} |")
        summary_lines.append(f"| {'Allocated memory peak':<35} | {format_value('allocated_bytes.all.peak'):>25} |")
        summary_lines.append(f"| {'Reserved memory current':<35} | {format_value('reserved_bytes.all.current'):>25} |")
        summary_lines.append(f"| {'Reserved memory peak':<35} | {format_value('reserved_bytes.all.peak'):>25} |")
        summary_lines.append(f"| {'Active memory current':<35} | {format_value('active_bytes.all.current'):>25} |")
        summary_lines.append(f"| {'Active memory peak':<35} | {format_value('active_bytes.all.peak'):>25} |")
        summary_lines.append(f"| {'Inactive split current':<35} | {format_value('inactive_split_bytes.all.current'):>25} |")
        summary_lines.append(f"| {'Inactive split peak':<35} | {format_value('inactive_split_bytes.all.peak'):>25} |")
        summary_lines.append("-----------------------------------------------------------------------")
        return "\n".join(summary_lines)

    return f"{device.type.upper()}: Memory summary not available."

def apply_patches() -> None:
    """Apply memory stats patches."""
    log_info("Applying memory stats patches")

    # Patch memory stats functions for CUDA
    if hasattr(torch.cuda, 'memory_allocated'):
        torch.cuda.memory_allocated = memory_allocated
    if hasattr(torch.cuda, 'memory_reserved'):
        torch.cuda.memory_reserved = memory_reserved
    if hasattr(torch.cuda, 'max_memory_allocated'):
        torch.cuda.max_memory_allocated = max_memory_allocated
    if hasattr(torch.cuda, 'max_memory_reserved'):
        torch.cuda.max_memory_reserved = max_memory_reserved
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats = reset_peak_memory_stats
    if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
        torch.cuda.reset_accumulated_memory_stats = reset_accumulated_memory_stats
    if hasattr(torch.cuda, 'mem_get_info'):
        torch.cuda.mem_get_info = mem_get_info
    if hasattr(torch.cuda, 'memory_stats'):
        torch.cuda.memory_stats = memory_stats
    if hasattr(torch.cuda, 'memory_summary'):
        torch.cuda.memory_summary = memory_summary

    log_info("Memory stats patches applied")


# Module initialization
log_info("Initializing TorchDevice memory stats module")

__all__: list[str] = [
    'memory_allocated',
    'memory_reserved',
    'max_memory_allocated',
    'max_memory_reserved',
    'reset_peak_memory_stats',
    'reset_accumulated_memory_stats',
    'mem_get_info',
    'memory_stats',
    'memory_summary',
    'apply_patches'
]

log_info("TorchDevice memory stats module initialized")
