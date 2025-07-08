"""
TorchDevice CPU Operations Module
-----------------------------
CPU-specific operations and patches.
"""

import torch
from ...core.logger import auto_log, log_info


# This module is currently a placeholder for any future CPU-specific operations.
# The primary numpy() patch is handled in core.tensors to avoid circular dependencies.

log_info("Initializing TorchDevice CPU operations module")

__all__: list[str] = []

log_info("TorchDevice CPU operations module initialized")