"""
TorchDevice RNG/Seed Logic
-------------------------
All random number generation and seed management logic for TorchDevice is centralized in this module.
This includes patching for torch and torch.cuda, and device-aware tensor creation wrappers.
"""

import torch
from typing import Callable, Any, List, Optional
from ..modules.TDLogger import auto_log
from ..TorchDevice import TorchDevice

# Store original functions before patching - these should only be used internally
t_manual_seed = torch.manual_seed
t_seed = torch.seed
t_get_rng_state = torch.get_rng_state
t_set_rng_state = torch.set_rng_state
t_initial_seed = torch.initial_seed if hasattr(torch, "initial_seed") else None

def tensor_creation_wrapper(original_func: Callable) -> Callable:
    """
    Wrapper for tensor creation functions to enforce default device redirection and CPU override.
    Always calls torch_device_replacement for the device argument, so the toggle state is always respected.
    """
    @auto_log()
    def wrapped_func(*args, **kwargs):
        device_arg = kwargs.get('device', None)
        if device_arg is None:
            device = TorchDevice.torch_device_replacement()
            kwargs['device'] = device
        else:
            device = TorchDevice.torch_device_replacement(device_arg)
            kwargs['device'] = device
        return original_func(*args, **kwargs)
    return wrapped_func

# Store patch status to avoid infinite recursion
_patched = False

def manual_seed(seed: int) -> None:
    """Set the random seed for PyTorch and MPS if available."""
    global _patched
    
    # Use original function directly when called internally
    if _patched:
        t_manual_seed(seed)
        if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(seed)
    else:
        # This call comes from user code, prevent recursion
        t_manual_seed(seed)
        if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(seed)

def manual_seed_all(seed: int) -> None:
    """Set the random seed for all devices (only one device in MPS/CPU)."""
    manual_seed(seed)

def seed() -> int:
    """Set a random seed for PyTorch and MPS if available, and return it."""
    s = t_seed()
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(s)
    return s

def seed_all() -> int:
    """Set a random seed for all devices (only one device in MPS/CPU)."""
    return seed()

def get_rng_state(device: Optional[Any] = None) -> torch.Tensor:
    """Get RNG state for the current device (MPS or CPU)."""
    return t_get_rng_state()

def set_rng_state(state: torch.Tensor, device: Optional[Any] = None) -> None:
    """Set RNG state for the current device (MPS or CPU)."""
    t_set_rng_state(state)

def get_rng_state_all() -> List[torch.Tensor]:
    """Get RNG state for all devices (only one device in MPS/CPU)."""
    return [t_get_rng_state()]

def set_rng_state_all(states: List[torch.Tensor]) -> None:
    """Set RNG state for all devices (only one device in MPS/CPU)."""
    if states:
        t_set_rng_state(states[0])

def initial_seed() -> int:
    """Return the initial seed for PyTorch."""
    if t_initial_seed:
        return t_initial_seed()
    return t_seed()

def apply_patches() -> None:
    """Apply all tensor creation and RNG-related patches."""
    global _patched
    # Only patch once
    if _patched:
        return
        
    # Patch tensor creation functions
    tensor_creation_functions = [
        'tensor', 'zeros', 'ones', 'empty', 'full',
        'arange', 'linspace', 'logspace', 'as_tensor',
        'empty_like', 'zeros_like', 'ones_like', 'full_like',
        'rand', 'randn', 'randint', 'rand_like', 'randn_like', 'randint_like'
    ]
    for func_name in tensor_creation_functions:
        if hasattr(torch, func_name):
            original_func = getattr(torch, func_name)
            patched_func = tensor_creation_wrapper(original_func)
            setattr(torch, func_name, patched_func)
    
    # Patch RNG/seed logic for torch and torch.cuda
    # Set flag before patching to avoid recursion
    _patched = True
    
    torch.manual_seed = t_manual_seed
    torch.seed = t_seed
    torch.get_rng_state = t_get_rng_state
    torch.set_rng_state = t_set_rng_state
    
    if hasattr(torch, "cuda"):
        # Wrap functions to match expected types
        # Type-annotated wrappers to fix lint errors
        def cuda_manual_seed(seed: int) -> None:
            return t_manual_seed(seed)
        torch.cuda.manual_seed = cuda_manual_seed

        def cuda_manual_seed_all(seed: int) -> None:
            return manual_seed_all(seed)
        torch.cuda.manual_seed_all = cuda_manual_seed_all

        def cuda_seed() -> None:
            t_seed()
        torch.cuda.seed = cuda_seed

        def cuda_seed_all() -> None:
            seed_all()
        torch.cuda.seed_all = cuda_seed_all

        def cuda_get_rng_state() -> torch.Tensor:
            return t_get_rng_state()
        torch.cuda.get_rng_state = cuda_get_rng_state

        def cuda_set_rng_state(state: torch.Tensor) -> None:
            t_set_rng_state(state)
        torch.cuda.set_rng_state = cuda_set_rng_state

        def cuda_get_rng_state_all() -> List[torch.Tensor]:
            return get_rng_state_all()
        torch.cuda.get_rng_state_all = cuda_get_rng_state_all

        def cuda_set_rng_state_all(states: List[torch.Tensor]) -> None:
            set_rng_state_all(states)
        torch.cuda.set_rng_state_all = cuda_set_rng_state_all

        def cuda_initial_seed() -> int:
            return initial_seed()
        torch.cuda.initial_seed = cuda_initial_seed