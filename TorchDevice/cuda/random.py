"""
TorchDevice RNG/Seed Logic
-------------------------
All random number generation and seed management logic for TorchDevice is centralized in this module.
This includes patching for torch and torch.cuda, and device-aware tensor creation wrappers.
"""

import torch
from typing import Callable, Any, List, Optional
from ..modules.TDLogger import auto_log, log_info
from ..TorchDevice import TorchDevice

def tensor_creation_wrapper(original_func: Callable) -> Callable:
    """
    Wrapper for tensor creation functions to enforce default device redirection and CPU override.
    Always calls torch_device_replacement for the device argument, so the toggle state is always respected.
    """
    @auto_log()
    def wrapped_func(*args, **kwargs):
        device_arg = kwargs.get('device', None)
        # If device is not specified, inject the current device (default or override)
        if device_arg is None:
            device = TorchDevice.torch_device_replacement()
            log_info(f"[tensor_creation_wrapper] Injecting device: {device}")
            kwargs['device'] = device
        else:
            # Always pass through torch_device_replacement to handle override logic
            device = TorchDevice.torch_device_replacement(device_arg)
            log_info(f"[tensor_creation_wrapper] Normalized device: {device}")
            kwargs['device'] = device
        return original_func(*args, **kwargs)
    return wrapped_func

# RNG/seed logic implementations

def manual_seed(seed: int) -> None:
    """Set the random seed for PyTorch and MPS if available."""
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

def manual_seed_all(seed: int) -> None:
    """Set the random seed for all devices (only one device in MPS/CPU)."""
    manual_seed(seed)

def seed() -> int:
    """Set a random seed for PyTorch and MPS if available, and return it."""
    s = torch.seed()
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(s)
    return s

def seed_all() -> int:
    """Set a random seed for all devices (only one device in MPS/CPU)."""
    return seed()

def get_rng_state(device: Optional[Any] = None) -> torch.Tensor:
    """Get RNG state for the current device (MPS or CPU)."""
    return torch.get_rng_state()

def set_rng_state(state: torch.Tensor, device: Optional[Any] = None) -> None:
    """Set RNG state for the current device (MPS or CPU)."""
    torch.set_rng_state(state)

def get_rng_state_all() -> List[torch.Tensor]:
    """Get RNG state for all devices (only one device in MPS/CPU)."""
    return [torch.get_rng_state()]

def set_rng_state_all(states: List[torch.Tensor]) -> None:
    """Set RNG state for all devices (only one device in MPS/CPU)."""
    if states:
        torch.set_rng_state(states[0])

def initial_seed() -> int:
    """Return the initial seed for PyTorch."""
    return torch.initial_seed() if hasattr(torch, "initial_seed") else torch.seed()

def apply_patches() -> None:
    # Patch tensor creation functions
    tensor_creation_functions = [
        'tensor', 'zeros', 'ones', 'empty', 'randn', 'rand', 'randint', 'arange', 'linspace', 'logspace'
    ]
    for func_name in tensor_creation_functions:
        if hasattr(torch, func_name):
            original_func = getattr(torch, func_name)
            setattr(torch, func_name, tensor_creation_wrapper(original_func))
    # Patch RNG/seed logic for torch and torch.cuda
    torch.manual_seed = manual_seed
    torch.seed = seed
    torch.get_rng_state = get_rng_state
    torch.set_rng_state = set_rng_state
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed = manual_seed
        torch.cuda.manual_seed_all = manual_seed_all
        torch.cuda.seed = seed
        torch.cuda.seed_all = seed_all
        torch.cuda.get_rng_state = get_rng_state
        torch.cuda.set_rng_state = set_rng_state
        torch.cuda.get_rng_state_all = get_rng_state_all
        torch.cuda.set_rng_state_all = set_rng_state_all
        torch.cuda.initial_seed = initial_seed 