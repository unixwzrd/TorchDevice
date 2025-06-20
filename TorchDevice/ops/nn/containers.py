"""
TorchDevice Neural Network Containers Module
---------------------------------------
Neural network container classes (Sequential, ModuleList, etc.).
"""

import torch
from TorchDevice.core.logger import log_info, auto_log


@auto_log()
def to_device(module: torch.nn.Module, device: torch.device = None) -> torch.nn.Module:
    """Move module and its parameters to the specified device."""
    from TorchDevice.core.device import DeviceManager  # Local import
    device = DeviceManager.torch_device_replacement(device)
    return module.to(device)


def apply_patches() -> None:
    """Apply neural network container patches."""
    from TorchDevice.core.device import DeviceManager  # Local import
    log_info("Applying nn container patches")

    # Store original to() method
    original_to = torch.nn.Module.to

    # Patch the to() method to use our device management
    def patched_to(self, *args, **kwargs):
        if args:
            # First argument is always the device spec if present
            device = DeviceManager.torch_device_replacement(args[0])
            args = (device,) + args[1:]
        elif 'device' in kwargs:
            # Or it might be in kwargs
            kwargs['device'] = DeviceManager.torch_device_replacement(kwargs['device'])
        return original_to(self, *args, **kwargs)

    torch.nn.Module.to = patched_to

    log_info("Neural network container patches applied")


# Module initialization
log_info("Initializing TorchDevice nn containers module")

__all__: list[str] = [
    'to_device',
    'apply_patches'
]

log_info("TorchDevice nn containers module initialized")