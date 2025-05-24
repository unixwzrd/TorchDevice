"""
TorchDevice Central Patch Application
------------------------------------
This module orchestrates the patching of all device, memory, random, and stub logic for TorchDevice.
Call apply_all_patches() to patch all relevant torch and torch.cuda functions for cross-backend compatibility.
"""

def apply_all_patches() -> None:
    from . import device, memory, random, streams, unassigned
    device.apply_patches()
    memory.apply_patches()
    random.apply_patches()
    streams.apply_patches()
    unassigned.apply_patches() 