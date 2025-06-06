"""
TorchDevice Core Patch Module
--------------------------
Core patching functionality and orchestration.
"""

import functools
import inspect
import logging
from typing import Callable, TypeVar, Any
import torch

from .logger import log_info, auto_log
from . import device # Import device directly
from . import tensors as core_tensors
from . import modules as core_modules
from .device import DeviceManager

# Define tensor_creation_wrapper before importing ops modules to avoid circular imports
T = TypeVar('T')

def tensor_creation_wrapper(func: Callable[..., T]) -> Callable[..., T]:
    """Wrapper for tensor creation functions to enforce default device redirection and CPU override.
    Always calls torch_device_replacement for the device argument, so the toggle state is always respected.
    The device object returned by DeviceManager.torch_device_replacement is used directly.
    """
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
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
        
        # Prepare extra info for logging
        frame = inspect.currentframe().f_back
        caller_info_extra = {
            'program_name': 'TorchDevice',
            'caller_func_name': frame.f_code.co_name if frame else 'unknown_func',
            'caller_filename': frame.f_code.co_filename if frame else 'unknown_file',
            'caller_lineno': frame.f_lineno if frame else 0,
            'torch_function': func.__name__
        }

        # Log if a redirection or significant change occurred.
        if device_arg_original is not None and str(device_arg_original) != str(final_device_obj):
            log_message = "GPU REDIRECT - Requested: %s -> Used: %s" % (str(device_arg_original), str(final_device_obj))
            logger = logging.getLogger("TorchDevice")
            logger.info(log_message, extra=caller_info_extra)
        # No explicit logging if device_arg_original was None, to match original behavior pattern.

        return func(*args, **kwargs)
    return wrapped_func

# Import operation modules after defining the wrapper
from TorchDevice.ops import (
    memory,
    nn,
    random,
    streams,
    events,
    autograd
)

# Import utility modules
from TorchDevice.utils import (
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
                # log_info(f"Patched torch.{func_name}") # Can be verbose, enable if needed
            else:
                log_info(f"Skipping torch.{func_name} as it's not callable.")
        else:
            log_info(f"Skipping torch.{func_name} as it does not exist.")
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
    from TorchDevice import ops 
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
    """Ensure all patches are applied."""
    apply_patches()


__all__: list[str] = [
    'apply_patches',
    'ensure_patched',
    'tensor_creation_wrapper'
]

log_info("TorchDevice core patch module initialized") 