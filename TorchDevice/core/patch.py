"""
TorchDevice Core Patch Module
--------------------------
Core patching functionality and orchestration.
"""

import functools
import inspect
import logging
import threading
from typing import Callable, TypeVar, Any
import torch

from .logger import log_info, auto_log
from . import device # Import device directly
from . import tensors as core_tensors
from . import modules as core_modules
from .device import DeviceManager

# Thread-local guard to prevent re-entry into the tensor creation wrapper
_in_tensor_creation_wrapper = threading.local()

# Define tensor_creation_wrapper before importing ops modules to avoid circular imports
T = TypeVar('T')

def tensor_creation_wrapper(func: Callable[..., T]) -> Callable[..., T]:
    """Wrapper for tensor creation functions to enforce default device redirection and CPU override.
    Always calls torch_device_replacement for the device argument, so the toggle state is always respected.
    The device object returned by DeviceManager.torch_device_replacement is used directly.
    """
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        # Initialize thread-local if not present
        if not hasattr(_in_tensor_creation_wrapper, 'value'):
            _in_tensor_creation_wrapper.value = False

        # If we are already in a tensor creation wrapper, pass through to avoid nested logging.
        if _in_tensor_creation_wrapper.value:
            return func(*args, **kwargs)
        
        _in_tensor_creation_wrapper.value = True
        try:
            # Get the original device argument (if any) for logging and processing
            device_arg_original = kwargs.get('device', None)
            
            final_device_obj: torch.device
            if device_arg_original is None:
                # No device specified by user, get the current effective device from DeviceManager
                final_device_obj = DeviceManager.torch_device_replacement()
            else:
                # User specified a device, let DeviceManager process it (handles 'cpu:-1', redirection, etc.)
                final_device_obj = DeviceManager.torch_device_replacement(device_arg_original)
            
            # Update kwargs with the definitively determined device object
            # This final_device_obj is the one that will be passed to the original PyTorch function
            kwargs['device'] = final_device_obj
            
            # The logging is now handled by the @auto_log decorator on torch_device_replacement.
            # The custom logging logic previously here was redundant and has been removed.

            return func(*args, **kwargs)
        finally:
            _in_tensor_creation_wrapper.value = False
    return wrapped_func

# Import operation modules after defining the wrapper
from ..ops import (
    memory,
    nn,
    random,
    streams,
    events,
    autograd
)

# Import utility modules
from ..utils import (
    compile,
    device_utils,
    error_handling,
    type_utils
)

log_info("Initializing TorchDevice core patch module")

# Track patch status
_core_patched = False
_ops_patched = False
_utils_patched = False

# Type variable for preserving function types
T = TypeVar('T', bound=Callable[..., Any])

_original_torch_creation_functions: dict[str, Callable] = {}

# List of torch tensor creation functions to be wrapped.
# These are top-level functions in the torch module that typically accept a 'device' kwarg.
_TENSOR_CREATION_FUNCTIONS_TO_WRAP = [
    'tensor', 'as_tensor',
    'scalar_tensor',
    'ones', 'zeros', 'empty', 'full', 'eye',
    'ones_like', 'zeros_like', 'empty_like', 'full_like',
    'arange', 'range', 'linspace', 'logspace', # range is torch.range, not python range
    'rand', 'randn', 'randint',
    'empty_strided',
    # Complex tensors
    'complex', 'polar',
    # Sparse tensors (ensure wrapper handles their specific args if different)
    # 'sparse_coo_tensor', 'sparse_csr_tensor', 'sparse_csc_tensor',
    # 'sparse_bsr_tensor', 'sparse_bsc_tensor', 'sparse_compressed_tensor',
    # '_sparse_coo_tensor_unsafe', '_sparse_csr_tensor_unsafe',
]

def _apply_core_patches() -> None:
    """Apply core functionality patches."""
    global _core_patched
    if _core_patched:
        log_info("Core patches already applied")
        return

    log_info("Applying core patches...")
    
    # 1. Apply device patches (torch.device, torch.load)
    log_info("  Applying device.apply_patches()...")
    device.apply_patches()
    
    # 2. Apply tensor method patches (Tensor.to, Tensor.cuda, etc.)
    log_info("  Applying core_tensors.apply_patches()...")
    core_tensors.apply_patches()

    # 3. Apply module method patches (Module.to, Module.cuda, etc.)
    log_info("  Applying core_modules.apply_patches()...")
    core_modules.apply_patches()
    
    # 4. Apply tensor creation function wrappers (torch.tensor, torch.ones, etc.)
    log_info("  Applying tensor creation function wrappers...")
    for func_name in _TENSOR_CREATION_FUNCTIONS_TO_WRAP:
        if hasattr(torch, func_name):
            original_func = getattr(torch, func_name)
            if callable(original_func):
                if func_name not in _original_torch_creation_functions:
                    _original_torch_creation_functions[func_name] = original_func
                
                wrapped_func = tensor_creation_wrapper(original_func)
                setattr(torch, func_name, wrapped_func)
                # log_info("Patched torch.%s", func_name) # Can be verbose, enable if needed
            else:
                log_info("Skipping torch.%s as it's not callable.", func_name)
        else:
            log_info("Skipping torch.%s as it does not exist.", func_name)
    log_info("  Tensor creation function wrappers applied.")

    _core_patched = True
    log_info("Core patches application complete.")


def _apply_ops_patches() -> None:
    """Apply operation-specific patches by calling the central apply_patches of the ops package."""
    global _ops_patched
    if _ops_patched:
        log_info("Operation patches already applied")
        return

    # Import ops package here to ensure it's fully initialized before calling its apply_patches
    from .. import ops 
    log_info("Applying all operation package patches via ops.apply_patches()")
    ops.apply_patches() # Call the apply_patches from TorchDevice/ops/__init__.py
    _ops_patched = True
    log_info("All operation package patches applied")


def _apply_utils_patches() -> None:
    """Apply utility patches."""
    global _utils_patched
    if _utils_patched:
        log_info("Utility patches already applied")
        return

    log_info("Applying utility patches")
    compile.apply_patches()
    device_utils.apply_patches()
    error_handling.apply_patches()
    type_utils.apply_patches()
    _utils_patched = True
    log_info("Utility patches applied")


def apply_patches() -> None:
    """
    Apply all TorchDevice patches in the correct order:
    1. Core patches (device, logging)
    2. Operation patches (memory, nn, random, streams, events, autograd)
    3. Utility patches (compile, device_utils, error_handling, type_utils)
    """
    log_info("Starting TorchDevice patch application")
    
    # 1. Core patches
    _apply_core_patches()
    
    # 2. Operation patches
    _apply_ops_patches()
    
    # 3. Utility patches
    _apply_utils_patches()
    
    log_info("TorchDevice patch application complete")



def ensure_patched() -> None:
    """Ensure that all patches are applied, but only once."""
    apply_patches()


__all__: list[str] = [
    'apply_patches',
    'ensure_patched',
    'tensor_creation_wrapper'
]

log_info("TorchDevice core patch module initialized") 