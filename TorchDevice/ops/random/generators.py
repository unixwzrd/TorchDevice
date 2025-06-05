"""
TorchDevice RNG/Seed Logic
-------------------------
All random number generation and seed management logic for TorchDevice is centralized in this module.
This includes patching for torch and torch.cuda, and device-aware tensor creation wrappers.
"""

import torch
import functools # Added functools for @functools.wraps
from typing import Callable, Any, List, Optional

from TorchDevice.core.logger import auto_log, log_info # Use actual logger
from TorchDevice.core.device import DeviceManager    # Use actual DeviceManager
from TorchDevice.core.patch import tensor_creation_wrapper # Use centralized wrapper
import TorchDevice.ops.device.cuda as cuda_device_ops # ADDED IMPORT

# Store original functions before patching - these should only be used internally
t_manual_seed = torch.manual_seed
t_seed = torch.seed
t_get_rng_state = torch.get_rng_state
t_set_rng_state = torch.set_rng_state
t_initial_seed: Optional[Callable[[], int]] = getattr(torch, "initial_seed", None)
original_torch_mps_manual_seed: Optional[Callable[[int], None]] = None
original_torch_mps_set_rng_state: Optional[Callable[[torch.Tensor], None]] = None

default_generator: Optional[Any] = None
_original_cuda_manual_seed: Optional[Callable[[int], None]] = None
_original_cuda_manual_seed_all: Optional[Callable[[int], None]] = None # Note: PyTorch's actual signature is seed_all(seed=None) but we capture it as callable without args for simplicity if it's just a pass-through
_original_cuda_seed: Optional[Callable[[int], None]] = None
_original_cuda_seed_all: Optional[Callable[[], None]] = None # PyTorch's actual signature is seed_all() -> None
_original_cuda_get_rng_state: Optional[Callable[[Optional[Any]], torch.Tensor]] = None
_original_cuda_set_rng_state: Optional[Callable[[torch.Tensor, Optional[Any]], None]] = None
_original_cuda_initial_seed: Optional[Callable[[], int]] = None
_original_cuda_get_rng_state_all: Optional[Callable[[], List[torch.Tensor]]] = None
_original_cuda_set_rng_state_all: Optional[Callable[[List[torch.Tensor]], None]] = None
original_torch_mps_get_rng_state: Optional[Callable[[], torch.Tensor]] = None

# Store patch status to avoid infinite recursion
_patched = False

def manual_seed(seed_val: int) -> None: # Renamed arg to avoid confusion with module-level 'seed' function
    """Set the random seed for PyTorch and MPS if available."""
    global _patched
    
    # Use original function directly when called internally by our patches
    if _patched:
        t_manual_seed(seed_val)
        if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
            # print(f"manual_seed (patched context): Seeding MPS with {seed_val}")
            torch.mps.manual_seed(seed_val)
    else:
        # This call comes from user code, or an unpatched part of torch itself.
        # We want to ensure this seeds the CPU and MPS if available, 
        # but NOT CUDA if this function is being used as a mock for torch.cuda.manual_seed.
        # print(f"manual_seed (user context): Seeding PyTorch (CPU) with {seed_val}")
        t_manual_seed(seed_val)
        if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
            # print(f"manual_seed (user context): Seeding MPS with {seed_val}")
            torch.mps.manual_seed(seed_val)

def manual_seed_all(seed_val: int) -> None:
    """Set the random seed for all devices (only one device in MPS/CPU context for this specific function)."""
    # This function, in the original random.py, was meant for non-CUDA contexts primarily.
    # It just calls our manual_seed, which handles CPU and MPS.
    # print(f"manual_seed_all: Calling local manual_seed with {seed_val}")
    manual_seed(seed_val)

def seed() -> int:
    """Set a random seed for PyTorch and MPS if available, and return it."""
    s = t_seed() # Calls original torch.seed()
    # print(f"seed: torch.seed() returned {s}")
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        # print(f"seed: Seeding MPS with {s}")
        torch.mps.manual_seed(s)
    return s

def seed_all() -> int:
    """Set a random seed for all devices (only one device in MPS/CPU context for this specific function)."""
    # print(f"seed_all: Calling local seed()")
    return seed()

def get_rng_state(device: Optional[Any] = None) -> torch.Tensor:
    """Get RNG state for the current device (MPS or CPU)."""
    # print(f"get_rng_state: Calling t_get_rng_state for device {device}")
    return t_get_rng_state()

def set_rng_state(state: torch.Tensor, device: Optional[Any] = None) -> None:
    """Set RNG state for the current device (MPS or CPU)."""
    # print(f"set_rng_state: Calling t_set_rng_state for device {device}")
    t_set_rng_state(state)

def get_rng_state_all() -> List[torch.Tensor]:
    """Get RNG state for all devices (only one device in MPS/CPU)."""
    # print(f"get_rng_state_all: Returning [t_get_rng_state()]")
    return [t_get_rng_state()]

def set_rng_state_all(states: List[torch.Tensor]) -> None:
    """Set RNG state for all devices (only one device in MPS/CPU)."""
    if states:
        # print(f"set_rng_state_all: Calling t_set_rng_state with states[0]")
        t_set_rng_state(states[0])

def initial_seed() -> int:
    """Return the initial seed for PyTorch."""
    if t_initial_seed:
        # print(f"initial_seed: Returning t_initial_seed()")
        return t_initial_seed()
    # print(f"initial_seed: t_initial_seed not available, returning t_seed()")
    return t_seed() # Fallback if torch.initial_seed doesn't exist (older PyTorch)

def _capture_original_rng_functions():
    global default_generator, t_manual_seed, t_initial_seed, t_get_rng_state, t_set_rng_state, t_seed
    global _original_cuda_manual_seed, _original_cuda_manual_seed_all, _original_cuda_seed, _original_cuda_seed_all
    global _original_cuda_get_rng_state, _original_cuda_set_rng_state, _original_cuda_initial_seed
    global _original_cuda_get_rng_state_all, _original_cuda_set_rng_state_all
    global original_torch_mps_manual_seed, original_torch_mps_get_rng_state, original_torch_mps_set_rng_state

    # Ensure capture only once
    # Check a few key captures; if they are already done, assume all are.
    if t_manual_seed is not None and default_generator is not None:
        # log_info("Original RNG functions already captured.") # Optional: can be noisy
        return

    log_info("Capturing original RNG functions for ops.random.generators.")
    default_generator = torch.default_generator
    t_manual_seed = torch.manual_seed
    t_initial_seed = getattr(torch, "initial_seed", None) # Handle if not present in older PyTorch
    t_get_rng_state = torch.get_rng_state
    t_set_rng_state = torch.set_rng_state
    t_seed = torch.seed

    # CUDA general RNG functions
    if hasattr(torch, 'cuda'):
        if hasattr(torch.cuda, 'manual_seed'):
            _original_cuda_manual_seed = torch.cuda.manual_seed
            log_info("Captured torch.cuda.manual_seed")
        if hasattr(torch.cuda, 'manual_seed_all'):
            _original_cuda_manual_seed_all = torch.cuda.manual_seed_all
            log_info("Captured torch.cuda.manual_seed_all")
        if hasattr(torch.cuda, 'seed'):
            _original_cuda_seed = torch.cuda.seed
            log_info("Captured torch.cuda.seed")
        if hasattr(torch.cuda, 'seed_all'):
            _original_cuda_seed_all = torch.cuda.seed_all
            log_info("Captured torch.cuda.seed_all")
        if hasattr(torch.cuda, 'initial_seed'):
            _original_cuda_initial_seed = torch.cuda.initial_seed
            log_info("Captured torch.cuda.initial_seed")
        if hasattr(torch.cuda, 'get_rng_state'):
            _original_cuda_get_rng_state = torch.cuda.get_rng_state
            log_info("Captured torch.cuda.get_rng_state")
        if hasattr(torch.cuda, 'set_rng_state'):
            _original_cuda_set_rng_state = torch.cuda.set_rng_state
            log_info("Captured torch.cuda.set_rng_state")
        if hasattr(torch.cuda, 'get_rng_state_all'):
            _original_cuda_get_rng_state_all = torch.cuda.get_rng_state_all
            log_info("Captured torch.cuda.get_rng_state_all")
        if hasattr(torch.cuda, 'set_rng_state_all'):
            _original_cuda_set_rng_state_all = torch.cuda.set_rng_state_all
            log_info("Captured torch.cuda.set_rng_state_all")

    # MPS RNG functions
    if hasattr(torch, 'mps'):
        if hasattr(torch.mps, 'manual_seed'):
            original_torch_mps_manual_seed = torch.mps.manual_seed
            log_info("Captured torch.mps.manual_seed")
        if hasattr(torch.mps, 'get_rng_state'):
            original_torch_mps_get_rng_state = torch.mps.get_rng_state
            log_info("Captured torch.mps.get_rng_state")
        if hasattr(torch.mps, 'set_rng_state'):
            original_torch_mps_set_rng_state = torch.mps.set_rng_state
            log_info("Captured torch.mps.set_rng_state")
    log_info("Finished capturing original RNG functions for ops.random.generators.")

def apply_patches() -> None:
    """Apply all tensor creation and RNG-related patches."""
    global _patched
    # Only patch once
    if _patched:
        return
    log_info("TorchDevice ops.random.generators: Starting patching process...")
    _capture_original_rng_functions() # Ensure all original functions are captured first
        
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
            log_info(f"TorchDevice ops.random.generators: Patching torch.{func_name}")
            patched_func = tensor_creation_wrapper(original_func)
            setattr(torch, func_name, patched_func)
    
    # Patch RNG/seed logic for torch and torch.cuda
    # Set flag before patching to ensure our custom functions are called by subsequent patches,
    # and to allow our custom functions to call the true original torch functions without recursing into themselves.
    _patched = True
    log_info("TorchDevice ops.random.generators: _patched set to True")
    
    # Patch top-level torch RNG functions to use our wrappers
    log_info("TorchDevice ops.random.generators: Patching torch.manual_seed, torch.seed, etc.")
    torch.manual_seed = manual_seed 
    torch.seed = seed
    torch.get_rng_state = get_rng_state
    torch.set_rng_state = set_rng_state
    if hasattr(torch, "initial_seed"):
        log_info("TorchDevice ops.random.generators: Patching torch.initial_seed")
        torch.initial_seed = initial_seed
    
    if hasattr(torch, "cuda"):
        log_info("TorchDevice ops.random.generators: torch.cuda found. Patching torch.cuda RNG functions.")
        
        def cuda_manual_seed(seed_val: int) -> None:
            log_info(f"TorchDevice ops.random.generators.cuda_manual_seed: Called with {seed_val=}.")
            if not cuda_device_ops.__is_torch_actually_compiled_with_cuda_value:
                log_info(f"  CUDA not compiled. Seeding CPU (via t_manual_seed).")
                t_manual_seed(seed_val)  # Original torch.manual_seed from this module
                cuda_device_ops._universal_lazy_init_replacement()
            elif _original_cuda_manual_seed: # Use locally captured original
                log_info(f"  CUDA compiled. Delegating to original torch.cuda.manual_seed.")
                _original_cuda_manual_seed(seed_val)
            else:
                log_info(f"  CUDA compiled but no original torch.cuda.manual_seed captured. Seeding CPU as fallback.")
                t_manual_seed(seed_val)
        torch.cuda.manual_seed = cuda_manual_seed

        def cuda_manual_seed_all(seed_val: int) -> None:
            log_info(f"TorchDevice ops.random.generators.cuda_manual_seed_all: Called with {seed_val=}.")
            if not cuda_device_ops.__is_torch_actually_compiled_with_cuda_value:
                log_info(f"  CUDA not compiled. Seeding CPU and MPS (if available) as fallback for cuda_manual_seed_all.")
                # Seed CPU directly using the captured default_generator
                if default_generator is not None:
                    log_info("Seeding CPU generator directly using captured torch.default_generator.manual_seed for cuda_manual_seed_all.")
                    default_generator.manual_seed(seed_val)
                else:
                    log_info("torch.default_generator not captured, cannot seed CPU generator directly for cuda_manual_seed_all.")

                # Seed MPS if available, using the captured original_torch_mps_manual_seed
                if torch.backends.mps.is_available(): # Check for MPS availability
                    if original_torch_mps_manual_seed is not None:
                        log_info("MPS is available, calling captured original_torch_mps_manual_seed for cuda_manual_seed_all.")
                        try:
                            original_torch_mps_manual_seed(seed_val)
                        except Exception as e:
                            log_info("Error calling captured original_torch_mps_manual_seed for cuda_manual_seed_all: %s", e)
                    else:
                        log_info("original_torch_mps_manual_seed not captured, cannot seed MPS for cuda_manual_seed_all.")
                else:
                    log_info("MPS not available, skipping MPS seeding for cuda_manual_seed_all.")
            elif _original_cuda_manual_seed_all: # Use locally captured original
                log_info(f"  CUDA compiled. Delegating to original torch.cuda.manual_seed_all.")
                _original_cuda_manual_seed_all(seed_val)
            else:
                log_info(f"  CUDA compiled but no original torch.cuda.manual_seed_all captured. Seeding CPU/MPS as fallback.")
                t_manual_seed(seed_val)
                if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                    torch.mps.manual_seed(seed_val)
        torch.cuda.manual_seed_all = cuda_manual_seed_all

        def cuda_seed() -> int:
            log_info(f"TorchDevice ops.random.generators.cuda_seed: Called.")
            if not cuda_device_ops.__is_torch_actually_compiled_with_cuda_value:
                log_info(f"  CUDA not compiled. Seeding CPU/MPS (via t_seed).")
                s = t_seed() # Original torch.seed() from this module
                if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                    torch.mps.manual_seed(s)
                cuda_device_ops._universal_lazy_init_replacement()
                return s
            elif _original_cuda_seed: # Use locally captured original
                log_info(f"  CUDA compiled. Delegating to original torch.cuda.seed.")
                return _original_cuda_seed()
            else:
                log_info(f"  CUDA compiled but no original torch.cuda.seed captured. Seeding CPU/MPS as fallback.")
                s = t_seed()
                if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                    torch.mps.manual_seed(s)
                return s
        torch.cuda.seed = cuda_seed

        def cuda_seed_all() -> int:
            log_info(f"TorchDevice ops.random.generators.cuda_seed_all: Called.")
            if not cuda_device_ops.__is_torch_actually_compiled_with_cuda_value:
                log_info(f"  CUDA not compiled. Seeding CPU/MPS (via t_seed for all).")
                s = t_seed() # Original torch.seed() from this module
                if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                    torch.mps.manual_seed(s)
                cuda_device_ops._universal_lazy_init_replacement()
                return s
            elif _original_cuda_seed_all: # Use locally captured original
                log_info(f"  CUDA compiled. Delegating to original torch.cuda.seed_all.")
                return _original_cuda_seed_all()
            else:
                log_info(f"  CUDA compiled but no original torch.cuda.seed_all captured. Seeding CPU/MPS as fallback.")
                s = t_seed()
                if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                    torch.mps.manual_seed(s)
                return s
        torch.cuda.seed_all = cuda_seed_all

        def cuda_get_rng_state(device: Optional[Any]=None) -> torch.Tensor:
            log_info(f"TorchDevice ops.random.generators.cuda_get_rng_state: Called with {device=}.")
            if not cuda_device_ops.__is_torch_actually_compiled_with_cuda_value:
                log_info(f"  CUDA not compiled. Getting CPU RNG state (via t_get_rng_state).")
                return t_get_rng_state() # Original torch.get_rng_state() from this module
            elif _original_cuda_get_rng_state: # Use locally captured original
                log_info(f"  CUDA compiled. Delegating to original torch.cuda.get_rng_state.")
                return _original_cuda_get_rng_state(device) 
            else:
                log_info(f"  CUDA compiled but no original torch.cuda.get_rng_state captured. Getting CPU RNG state as fallback.")
                return t_get_rng_state()
        torch.cuda.get_rng_state = cuda_get_rng_state

        def cuda_set_rng_state(state_val: torch.Tensor, device: Optional[Any]=None) -> None:
            log_info(f"TorchDevice ops.random.generators.cuda_set_rng_state: Called with {device=}.")
            if not cuda_device_ops.__is_torch_actually_compiled_with_cuda_value:
                log_info(f"  CUDA not compiled. Setting CPU RNG state (via t_set_rng_state).")
                t_set_rng_state(state_val) # Original torch.set_rng_state() from this module
            elif _original_cuda_set_rng_state: # Use locally captured original
                log_info(f"  CUDA compiled. Delegating to original torch.cuda.set_rng_state.")
                _original_cuda_set_rng_state(state_val, device)
            else:
                log_info(f"  CUDA compiled but no original torch.cuda.set_rng_state captured. Setting CPU RNG state as fallback.")
                t_set_rng_state(state_val)
        torch.cuda.set_rng_state = cuda_set_rng_state

        def cuda_get_rng_state_all() -> List[torch.Tensor]:
            log_info(f"TorchDevice ops.random.generators.cuda_get_rng_state_all: Called.")
            if not cuda_device_ops.__is_torch_actually_compiled_with_cuda_value:
                log_info(f"  CUDA not compiled. Returning list with CPU RNG state.")
                states = [t_get_rng_state()] # CPU state from this module
                return states
            elif _original_cuda_get_rng_state_all: # Use locally captured original
                log_info(f"  CUDA compiled. Delegating to original torch.cuda.get_rng_state_all.")
                return _original_cuda_get_rng_state_all()
            else: 
                log_info(f"  CUDA compiled but no original torch.cuda.get_rng_state_all captured. Returning CPU RNG state list as fallback.")
                return [t_get_rng_state()]
        torch.cuda.get_rng_state_all = cuda_get_rng_state_all

        def cuda_set_rng_state_all(states_val: List[torch.Tensor]) -> None:
            log_info(f"TorchDevice ops.random.generators.cuda_set_rng_state_all: Called.")
            if not cuda_device_ops.__is_torch_actually_compiled_with_cuda_value:
                log_info(f"  CUDA not compiled. Setting CPU RNG state from first state in list.")
                if states_val:
                    t_set_rng_state(states_val[0]) # Set CPU state from this module
            elif _original_cuda_set_rng_state_all: # Use locally captured original
                log_info(f"  CUDA compiled. Delegating to original torch.cuda.set_rng_state_all.")
                _original_cuda_set_rng_state_all(states_val)
            else: 
                log_info(f"  CUDA compiled but no original torch.cuda.set_rng_state_all captured. Setting CPU RNG state as fallback.")
                if states_val:
                    t_set_rng_state(states_val[0])
        torch.cuda.set_rng_state_all = cuda_set_rng_state_all

        if hasattr(torch.cuda, "initial_seed"):
            log_info(f"TorchDevice ops.random.generators: Patching torch.cuda.initial_seed")
            def cuda_initial_seed() -> int:
                log_info(f"TorchDevice ops.random.generators.cuda_initial_seed: Called.")
                if not cuda_device_ops.__is_torch_actually_compiled_with_cuda_value:
                    log_info(f"  CUDA not compiled. Returning CPU initial seed (via t_initial_seed).")
                    return t_initial_seed() if t_initial_seed else 0 # t_initial_seed from this module
                elif _original_cuda_initial_seed: # Use locally captured original
                    log_info(f"  CUDA compiled. Delegating to original torch.cuda.initial_seed.")
                    return _original_cuda_initial_seed()
                else:
                    log_info(f"  CUDA compiled but no original torch.cuda.initial_seed captured. Returning CPU initial seed as fallback.")
                    return t_initial_seed() if t_initial_seed else 0
            torch.cuda.initial_seed = cuda_initial_seed
    log_info("TorchDevice ops.random.generators: apply_patches complete.")

__all__ = [
    'manual_seed',
    'manual_seed_all',
    'seed',
    'seed_all',
    'get_rng_state',
    'set_rng_state',
    'get_rng_state_all',
    'set_rng_state_all',
    'initial_seed',
    'apply_patches'
]

# Call apply_patches when the module is loaded to ensure patches are active.
# However, this might be too early if other modules are not yet loaded.
# Consider calling this from a central initialization point in TorchDevice's __init__.py
# For now, let's keep it here for simplicity during refactoring.
# apply_patches() 