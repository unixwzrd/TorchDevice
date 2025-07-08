"""
TorchDevice Utilities Module
------------------------
Utility functions and helpers.
"""

from ..core.logger import log_info, auto_log
from . import (
    compile as compile_module,
    device_utils as device_utils_module,
    error_handling as error_handling_module,
    profiling as profiling_module,
    type_utils as type_utils_module
)

# Import specific utility functions for easier access
from .device_utils import (
    is_cuda_effectively_available,
    is_mps_effectively_available,
    get_current_device_type,
    is_cpu_override_active
)

log_info("Initializing TorchDevice utils module")

@auto_log()
def apply_patches() -> None:
    """Apply patches from all utility submodules."""
    log_info("Applying utils patches")
    compile_module.apply_patches()
    device_utils_module.apply_patches() # Though it currently does nothing
    error_handling_module.apply_patches() # Though it currently does nothing
    # profiling_module.apply_patches() # profiling.py is currently empty
    type_utils_module.apply_patches() # Though it currently does nothing
    log_info("Utils patches applied")

__all__: list[str] = [
    # Export utility functions directly
    'is_cuda_effectively_available',
    'is_mps_effectively_available',
    'get_current_device_type',
    'is_cpu_override_active',
    # Export submodules for qualified access if needed
    'compile_module',
    'device_utils_module',
    'error_handling_module',
    # 'profiling_module', # profiling.py is currently empty
    'type_utils_module',
    # Export apply_patches for this module
    'apply_patches'
]

log_info("TorchDevice utils module initialized")