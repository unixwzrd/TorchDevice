"""
TorchDevice._deferred_patches - Run patches that need to be applied after main initialization

This module is imported at the end of __init__.py to ensure that these patches run
after all the core initialization is complete. This prevents circular import issues
and helps avoid premature trigger of imports that could cause initialization problems.
"""

from .modules.TDLogger import log_info
from .modules import compile

# Apply deferred patches
log_info("Applying deferred patches for PyTorch compiler")
compile.patch_dynamo_config()
log_info("Deferred patching complete") 