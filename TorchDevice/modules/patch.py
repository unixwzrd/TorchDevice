"""
TorchDevice Central Patch Application
-------------------------------------
Orchestrate all TorchDevice patches (device, memory, RNG, streams)
and disable Dynamo’s placeholder‐tests entirely on MPS.
"""

def apply_all_patches() -> None:
    # 0) If running on MPS, turn Dynamo’s placeholder‐test into a no-op
    import torch
    from ..TorchDevice import TorchDevice
    if TorchDevice.get_default_device() == 'mps':
        try:
            from torch._dynamo.variables import torch_function as _tf
            _tf.populate_builtin_to_tensor_fn_map = lambda: None
        except ImportError:
            # Dynamo not present or too old—nothing to do
            pass

    # 1) core patches—these must come first
    from ..device import device, memory, random, streams, unassigned
    device.apply_patches()
    memory.apply_patches()
    random.apply_patches()
    streams.apply_patches()
    unassigned.apply_patches()

    # 2) now install the compile/Dynamo patches if not pure CPU
    from ..TorchDevice import TorchDevice as _TD
    if _TD.get_default_device() != torch.device('cpu'):
        # temporarily force all torch.device(...) calls to pick the MPS device
        _orig = _TD.torch_device_replacement
        _TD.torch_device_replacement = lambda *a, **k: _TD.get_default_device()
        try:
            from ..modules import compile
            compile.apply_patches()
        finally:
            _TD.torch_device_replacement = _orig