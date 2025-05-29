"""
TorchDevice Random Generators Module
-----------------------------
Random number generation implementations.
"""

import torch
from typing import List, Optional, Callable
from ...core.logger import log_info, auto_log
from ...core.device import torch_device_replacement

log_info("Importing TorchDevice/ops/random/generators.py")

# Store original functions before patching - these should only be used internally
t_manual_seed = torch.manual_seed
t_seed = torch.seed
t_get_rng_state = torch.get_rng_state
t_set_rng_state = torch.set_rng_state
t_initial_seed = torch.initial_seed if hasattr(torch, "initial_seed") else None

# Store original CUDA functions if available
t_cuda_manual_seed = getattr(torch.cuda, 'manual_seed', None)
t_cuda_manual_seed_all = getattr(torch.cuda, 'manual_seed_all', None)
t_cuda_seed = getattr(torch.cuda, 'seed', None)
t_cuda_get_rng_state = getattr(torch.cuda, 'get_rng_state', None)
t_cuda_set_rng_state = getattr(torch.cuda, 'set_rng_state', None)
t_cuda_initial_seed = getattr(torch.cuda, 'initial_seed', None)

# Store original MPS functions if available
t_mps_manual_seed = getattr(torch.mps, 'manual_seed', None) if hasattr(torch, 'mps') else None
t_mps_seed = getattr(torch.mps, 'seed', None) if hasattr(torch, 'mps') else None
t_mps_get_rng_state = getattr(torch.mps, 'get_rng_state', None) if hasattr(torch, 'mps') else None
t_mps_set_rng_state = getattr(torch.mps, 'set_rng_state', None) if hasattr(torch, 'mps') else None

# Flag to prevent recursion in patched functions
_patched = False

@auto_log()
def _detect_default_device_type() -> str:
    """Detect the default device type based on available hardware."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

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
            device = torch_device_replacement()
            kwargs['device'] = device
        else:
            # Always pass through torch_device_replacement to handle override logic
            device = torch_device_replacement(device_arg)
            kwargs['device'] = device
        return original_func(*args, **kwargs)
    return wrapped_func

def _manual_seed(seed: int) -> None:
    """Set the seed for random number generation."""
    global _patched
    if not _patched:
        _patched = True
        try:
            t_manual_seed(seed)
            if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                t_mps_manual_seed(seed)
            if hasattr(torch.cuda, "manual_seed"):
                t_cuda_manual_seed(seed)
        finally:
            _patched = False

def _seed() -> int:
    """Generate a random seed."""
    return t_seed()

def _get_rng_state() -> torch.Tensor:
    """Get the current random number generator state."""
    return t_get_rng_state()

def _set_rng_state(state: torch.Tensor) -> None:
    """Set the random number generator state."""
    t_set_rng_state(state)

def _initial_seed() -> Optional[int]:
    """Get the initial seed used for random number generation."""
    return t_initial_seed() if t_initial_seed else None

def apply_patches() -> None:
    """Apply all tensor creation and RNG-related patches."""
    global _patched
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
    
    # Patch RNG/seed functions
    torch.manual_seed = _manual_seed
    torch.seed = _seed
    torch.get_rng_state = _get_rng_state
    torch.set_rng_state = _set_rng_state
    if t_initial_seed:
        torch.initial_seed = _initial_seed

    # Set flag after patching to avoid recursion
    _patched = True

__all__: List[str] = [
    'apply_patches'
]