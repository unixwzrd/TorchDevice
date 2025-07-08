from typing import Callable, Any, Set
from .logger import log_info

# Set of fully qualified function names to exclude from patching.
_PATCH_EXCLUSIONS: Set[str] = {
    'torch.nn.utils.rnn.pack_padded_sequence',
}


def patch_function(target_module: Any, func_name: str, wrapper: Callable) -> None:
    """Patches a function on a target module, respecting the exclusion list.

    Args:
        target_module: The module or class to patch.
        func_name: The name of the function/method to patch.
        wrapper: The wrapper function to replace the original with.
    """
    if not hasattr(target_module, func_name):
        log_info(f"Skipping patch for {func_name} in {target_module.__name__}: not found.")
        return

    original_func = getattr(target_module, func_name)
    full_func_name = f"{target_module.__name__}.{func_name}"

    if full_func_name in _PATCH_EXCLUSIONS:
        log_info(f"Skipping patch for {full_func_name}: in exclusion list.")
        return

    wrapped_func = wrapper(original_func)
    setattr(target_module, func_name, wrapped_func)
    log_info(f"Successfully patched {full_func_name}.")
