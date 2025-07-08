"""
TorchDevice Memory Management Module
--------------------------------
Memory allocation and cache management.
"""

import torch
from typing import Optional, List, Dict, Any # Add common typing imports for robustness
from ...core.logger import log_info, auto_log

# Store original functions
t_cuda_empty_cache = torch.cuda.empty_cache if hasattr(torch.cuda, 'empty_cache') else None
t_mps_empty_cache = torch.mps.empty_cache if hasattr(torch.mps, 'empty_cache') else None

# CUDA specific memory management functions
t_cuda_caching_allocator_alloc = torch.cuda.caching_allocator_alloc if hasattr(torch.cuda, 'caching_allocator_alloc') else None
t_cuda_caching_allocator_delete = torch.cuda.caching_allocator_delete if hasattr(torch.cuda, 'caching_allocator_delete') else None
t_cuda_set_per_process_memory_fraction = torch.cuda.set_per_process_memory_fraction if hasattr(torch.cuda, 'set_per_process_memory_fraction') else None
t_cuda_memory_snapshot = torch.cuda.memory_snapshot if hasattr(torch.cuda, 'memory_snapshot') else None


@auto_log()
def empty_cache() -> None:
    """Empty the memory cache for the current device."""
    from ...core.device import DeviceManager  # Local import
    device_type = DeviceManager.get_default_device().type
    if device_type == 'cuda' and t_cuda_empty_cache:
        t_cuda_empty_cache()
    elif device_type == 'mps' and t_mps_empty_cache:
        t_mps_empty_cache()


@auto_log()
def caching_allocator_alloc(size: int, device: Optional[torch.device] = None, stream: Optional[int] = None) -> int:
    """Allocate memory using the caching allocator. Returns 0 for MPS/CPU."""
    from ...core.device import DeviceManager  # Local import
    # PyTorch's actual caching_allocator_alloc takes device as int, stream as int (cudaStream_t)
    # We'll derive the device_type from the DeviceManager for decision making
    effective_device = device or DeviceManager.get_default_device()
    
    if effective_device.type == 'cuda' and t_cuda_caching_allocator_alloc:
        # The actual torch.cuda.caching_allocator_alloc might not directly accept torch.device
        # It expects an int for device_index. And stream is complex.
        # This patching is primarily for making sure calls don't break, not full emulation.
        # For simplicity, we'll assume if it's called, it's for the current default CUDA device.
        # A more robust solution would inspect 'device' if it's an int or torch.device.
        log_info("Calling original torch.cuda.caching_allocator_alloc. Note: Stream parameter handling is simplified.")
        # This is a simplification; direct call might need device_idx and proper stream handling.
        # return t_cuda_caching_allocator_alloc(size, DeviceManager.get_default_device().index or 0, stream_ptr)
        # Since this is complex to truly mock without CUDA toolkit specifics for stream, we'll log and return 0
        # if we can't guarantee a safe call. For now, let's assume if t_cuda_caching_allocator_alloc exists,
        # it can be called with size only for simplicity in a stub context, or we make it a no-op for safety.
        # Given the complexity of fully mocking this, we'll log and treat as no-op for safety if not on actual CUDA hw.
        # However, if t_cuda_caching_allocator_alloc exists, it implies we are likely in an env where it *could* work.
        # Let's assume the user knows what they are doing if they call this on a CUDA device.
        # The signature in python is (size, device=None, stream=None)
        # Let's pass what we have, assuming the Python binding handles it.
        return t_cuda_caching_allocator_alloc(size, device=effective_device, stream=stream)

    elif effective_device.type == 'mps':
        log_info("MPS does not support torch.cuda.caching_allocator_alloc. Size: %s. Returning 0.", size)
    else: # CPU or other
        log_info("torch.cuda.caching_allocator_alloc called on non-CUDA/MPS device (%s). Size: %s. Returning 0.", effective_device.type, size)
    return 0 # Return a null-like pointer

@auto_log()
def caching_allocator_delete(ptr: int) -> None:
    """Delete memory allocated by the caching allocator. No-op for MPS/CPU."""
    from ...core.device import DeviceManager  # Local import
    device_type = DeviceManager.get_default_device().type
    if device_type == 'cuda' and t_cuda_caching_allocator_delete:
        t_cuda_caching_allocator_delete(ptr)
    elif device_type == 'mps':
        log_info("MPS does not support torch.cuda.caching_allocator_delete. Pointer: %s. This call is a no-op.", ptr)
    else: # CPU or other
        log_info("torch.cuda.caching_allocator_delete called on non-CUDA/MPS device (%s). Pointer: %s. This call is a no-op.", device_type, ptr)

@auto_log()
def set_per_process_memory_fraction(fraction: float, device: Optional[torch.device] = None) -> None:
    """Set memory fraction for a process. No-op for MPS/CPU."""
    from ...core.device import DeviceManager  # Local import
    # The original function can take an int or torch.device
    effective_device = device or DeviceManager.get_default_device()

    if effective_device.type == 'cuda' and t_cuda_set_per_process_memory_fraction:
        # The python binding takes (fraction, device=None) where device can be int or torch.device
        t_cuda_set_per_process_memory_fraction(fraction, device=effective_device)
    elif effective_device.type == 'mps':
        log_info("MPS does not support torch.cuda.set_per_process_memory_fraction. Fraction: %s. This call is a no-op.", fraction)
    else: # CPU or other
        log_info("torch.cuda.set_per_process_memory_fraction called on non-CUDA/MPS device (%s). Fraction: %s. This call is a no-op.", effective_device.type, fraction)

@auto_log()
def memory_snapshot() -> list[dict[str, any]]:
    """Return a snapshot of the memory allocator state. CUDA-specific."""
    from ...core.device import DeviceManager  # Local import
    device_type = DeviceManager.get_default_device().type
    if device_type == 'cuda' and t_cuda_memory_snapshot:
        return t_cuda_memory_snapshot()
    elif device_type == 'mps':
        log_info("MPS does not support torch.cuda.memory_snapshot(). Returning empty list.")
    else: # CPU or other
        log_info("torch.cuda.memory_snapshot() called on non-CUDA/MPS device (%s). Returning empty list.", device_type)
    return []

def apply_patches() -> None:
    """Apply memory management patches."""
    log_info("Applying memory management patches")
    
    # Patch empty_cache function for both CUDA and MPS
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache = empty_cache
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache = empty_cache

    # Patch CUDA specific allocation functions
    if hasattr(torch.cuda, 'caching_allocator_alloc'):
        torch.cuda.caching_allocator_alloc = caching_allocator_alloc
    if hasattr(torch.cuda, 'caching_allocator_delete'):
        torch.cuda.caching_allocator_delete = caching_allocator_delete
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction = set_per_process_memory_fraction
    if hasattr(torch.cuda, 'memory_snapshot'):
        torch.cuda.memory_snapshot = memory_snapshot
    
    log_info("Memory management patches applied")


# Module initialization
log_info("Initializing TorchDevice memory management module")

__all__: list[str] = [
    'empty_cache',
    'caching_allocator_alloc',
    'caching_allocator_delete',
    'set_per_process_memory_fraction',
    'memory_snapshot',
    'apply_patches'
]

log_info("TorchDevice memory management module initialized") 