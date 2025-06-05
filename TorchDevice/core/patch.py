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
from . import device, tensors
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


def _apply_core_patches() -> None:
    """Apply core functionality patches."""
    global _core_patched
    if _core_patched:
        log_info("Core patches already applied")
        return

    log_info("Applying core patches")
    device.apply_patches()
    tensors.apply_patches()
    _core_patched = True
    log_info("Core patches applied")


def _apply_ops_patches() -> None:
    """Apply operation-specific patches."""
    global _ops_patched
    if _ops_patched:
        log_info("Operation patches already applied")
        return

    log_info("Applying operation patches")
    # Order matters here - device patches should be applied first,
    # followed by random before memory
    
    # Apply device-specific patches (CUDA/MPS)
    from TorchDevice.ops.device import cuda
    log_info("Applying CUDA device patches")
    cuda.apply_patches()
    
    log_info("Applying random patches")
    random.apply_patches()
    log_info("Applying memory patches")
    memory.apply_patches()
    log_info("Applying neural network patches")
    nn.apply_patches()
    log_info("Applying stream patches")
    streams.apply_patches()
    log_info("Applying event patches")
    events.apply_patches()
    log_info("Applying autograd patches")
    autograd.apply_patches()
    _ops_patched = True
    log_info("Operation patches applied")


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