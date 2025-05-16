"""
TorchDevice.modules.compile - Smart PyTorch compiler management for different devices

This module provides device-aware configuration of PyTorch's compilation features:

1. TorchDynamo (the torch.compile tracer):
   - Works on all devices (CPU, CUDA, MPS)
   - Kept enabled but configured appropriately for each device

2. TorchInductor (the default backend):
   - Only works properly on CUDA and CPU devices
   - For MPS devices, we configure to use 'aot_eager' backend instead
"""

from typing import Any
from .TDLogger import log_info


def _device_aware_compile(model: Any, *args: Any, **kwargs: Any) -> Any:
    """
    A device-aware wrapper for torch.compile that selects appropriate backends.
    
    On CUDA: Uses the default 'inductor' backend (or whatever was specified)
    On MPS: Switches to 'aot_eager' backend which works better on Metal
    On CPU: Uses the default backend (typically 'inductor')
    """
    # Import here to avoid circular imports
    import torch
    
    # Get the current device being used
    curr_device = None
    
    # Try to get current device from model if possible
    if hasattr(model, 'parameters'):
        try:
            params = list(model.parameters())
            if params and hasattr(params[0], 'device'):
                curr_device = params[0].device
        except Exception:
            pass
    
    # If we couldn't get device from model, check global default device
    if curr_device is None:
        try:
            # Import the main package rather than using relative imports
            import TorchDevice
            curr_device = TorchDevice.TorchDevice.get_default_device()
        except Exception:
            # Fall back to CPU as the safest option
            curr_device = torch.device('cpu')
    
    # Check if we're on MPS
    is_mps = (hasattr(curr_device, 'type') and curr_device.type == 'mps')
    
    # For MPS, use 'aot_eager' backend which works better
    if is_mps and 'backend' not in kwargs:
        log_info("Using 'aot_eager' backend for torch.compile on MPS device")
        kwargs['backend'] = 'aot_eager'
    
    # Call the original compile function with our adjusted arguments
    original_compile = getattr(_device_aware_compile, 't_compile', None)
    if original_compile:
        try:
            return original_compile(model, *args, **kwargs)
        except Exception as e:
            log_info(f"Error in torch.compile: {e}")
            log_info("Returning uncompiled model")
            return model
    else:
        # Original compile wasn't saved - return model uncompiled
        log_info("Original torch.compile not found, returning uncompiled model")
        return model


def patch_compile() -> None:
    """
    Patch the torch.compile function for device-aware operation.
    This is separate from _dynamo config patching to avoid import issues.
    """
    import torch
    
    # Save the original compile function
    if hasattr(torch, "compile"):
        _original = torch.compile
        _device_aware_compile.t_compile = _original  # type: ignore
        
        # Replace the compile function
        torch.compile = _device_aware_compile  # type: ignore
        log_info("Patched torch.compile with device-aware version")


def patch_dynamo_config() -> None:
    """
    Carefully configure torch._dynamo if it's already loaded.
    This is called separately to avoid forcing _dynamo to load too early.
    """
    import torch
    
    # ONLY get _dynamo if it's already loaded/imported
    if "_dynamo" in torch.__dict__:
        dynamo = torch._dynamo
        try:
            # Configure _dynamo in a way that works across devices
            if hasattr(dynamo, "config"):
                # Enable dynamic shapes which helps with different devices
                dynamo.config.dynamic_shapes = True
                # Don't try to use cudagraphs on non-CUDA devices
                dynamo.config.automatic_dynamic_shapes = True
                # Increase tolerance for numerical differences between devices
                dynamo.config.tolerance_for_precision = 1e-4
                log_info("Configured torch._dynamo for cross-device compatibility")
        except Exception as e:
            log_info(f"Error configuring torch._dynamo: {e}")


def apply_patches() -> None:
    """
    Apply patches to optimize PyTorch's compilation system for the current device.
    
    This patches:
    1. torch.compile - to use the appropriate backend based on device
    2. _dynamo config - to safely configure torch._dynamo if it's loaded
    """
    log_info("Applying PyTorch compiler patches")
    
    # Patch torch.compile first
    patch_compile()
    
    # We'll patch _dynamo config LAST, after all other patches are applied
    # to avoid triggering imports too early
    
    log_info("Primary compiler patching complete - _dynamo patching deferred until later") 