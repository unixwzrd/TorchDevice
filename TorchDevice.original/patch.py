"""
TorchDevice Central Patch Application
-------------------------------------
Orchestrate all TorchDevice patches (device, memory, RNG, streams, compile/Dynamo, and deferred patches).
This is the single entry point for all patching logic.
"""

import torch
from .device import device, memory, random, streams, unassigned
from .modules import compile as compile_mod
from .modules.TDLogger import log_info


def apply_all_patches() -> None:
    """
    Apply all TorchDevice monkey-patches in the correct order.
    This includes core patches, compile/Dynamo patches, and deferred patches.
    """
    log_info("[TorchDevice] patch.apply_all_patches called")
    # 1. Core patches
    device.apply_patches()
    memory.apply_patches()
    random.apply_patches()
    streams.apply_patches()
    unassigned.apply_patches()

    # 2. Compile/Dynamo patches (if not pure CPU)
    from .TorchDevice import TorchDevice as _TD
    if _TD.get_default_device() != torch.device('cpu'):
        # Temporarily force all torch.device(...) calls to pick the default accelerator
        orig = _TD.torch_device_replacement
        _TD.torch_device_replacement = lambda *a, **k: _TD.get_default_device()
        try:
            compile_mod.apply_patches()
        finally:
            _TD.torch_device_replacement = orig

    # 3. Deferred patches (previously in _deferred_patches.py)
    compile_mod.patch_dynamo_config()
    log_info("Deferred patching complete") 