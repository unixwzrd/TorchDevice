"""
TorchDevice Core Device Module
---------------------------
Core device functionality and patching.
"""

import threading
import torch
from .logger import log_info, log_error, auto_log
from . import hardware_info # Use our new hardware_info module

# Capture the original torch.device type
_ORIGINAL_TORCH_DEVICE_CLASS = torch.device("cpu").__class__


class PatchedTorchDeviceMeta(type):
    def __instancecheck__(self, instance):
        return isinstance(instance, _ORIGINAL_TORCH_DEVICE_CLASS)

    # Optional: if issubclass checks are also needed against the original type
    # def __subclasscheck__(cls, subclass):
    #     return issubclass(subclass, _ORIGINAL_TORCH_DEVICE_CLASS)

    # If torch.device had class attributes/methods that need to be accessible via PatchedTorchDevice
    # they could be proxied here. For torch.device, this is likely not an issue as it's mainly a constructor.
    # Example (use with caution, might conflict with type's own attributes):
    # def __getattr__(cls, name):
    #     return getattr(_ORIGINAL_TORCH_DEVICE_CLASS, name)



class PatchedTorchDevice(metaclass=PatchedTorchDeviceMeta):
    def __new__(cls, *args, **kwargs):
        return DeviceManager.torch_device_replacement(*args, **kwargs)



class DeviceManager:
    """Core device management functionality."""
    _default_device = None              # Actual torch.device type
    _default_device_type = ""           # Device type name (cuda, mps, cpu, etc...)
    _previous_default_device_type = ""  # Previous default device before CPU override
    _lock = threading.RLock()
    _cpu_override = False               # Flag for explicit CPU override

    # Store original torch functions
    t_device = torch.device

    @classmethod
    @auto_log()
    def get_default_device(cls):
        """Get the default device for tensor operations."""
        if cls._default_device is None:
            cls._default_device_type = cls._detect_default_device_type()
            cls._default_device = cls.t_device(cls._default_device_type)
        return cls._default_device

    @classmethod
    def cpu_override(cls):
        """Check if CPU override is active."""
        return cls._cpu_override

    @classmethod
    @auto_log()
    def _detect_default_device_type(cls):
        """Detect the best available device type."""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            cls._default_device_type = 'mps'
        elif torch.cuda.is_available() or torch.backends.cuda.is_built():
            cls._default_device_type = 'cuda'
        else:
            cls._default_device_type = 'cpu'
        return cls._default_device_type

    @classmethod
    @auto_log()
    def torch_device_replacement(cls, *args, **kwargs) -> torch.device:
        """
        Drop-in replacement for torch.device() with device redirection and CPU override toggle.
        • No arguments → returns default device (or CPU if override is active).
        • 'cpu:-1' or torch.device('cpu', -1) → toggle CPU override.
        • Redirects non-CPU devices to available hardware.
        • Preserves extra args and kwargs.
        Always returns a torch.device object.
        """
        device_type = ""
        device_index = None
        with cls._lock:
            # If first argument is torch.device, check for override
            if args and isinstance(args[0], _ORIGINAL_TORCH_DEVICE_CLASS):
                return args[0]

            # If first argument is string device spec, parse and modify
            if args and isinstance(args[0], str):
                device_spec = args[0].strip()
                if ":" in device_spec:
                    parts = device_spec.split(":", 1)
                    device_type = parts[0].lower()
                    try:
                        device_index = int(parts[1])
                    except ValueError:
                        device_index = None
                else:
                    device_type = device_spec.lower()

            # CPU override toggle logic
            if device_type == "cpu":
                if device_index == -1:
                    device_index = None
                    if cls.cpu_override():
                        # Toggle OFF
                        cls._cpu_override = False
                        cls._default_device_type = cls._previous_default_device_type
                        cls._previous_default_device_type = ""
                    else:
                        # Toggle ON
                        cls._cpu_override = True
                        cls._previous_default_device_type = cls._default_device_type
                        cls._default_device_type = 'cpu'

            device_type = cls._default_device_type
            result = cls.t_device(device_type, device_index)
            return result


class _TorchDevicePatcher:
    def __call__(self, *args, **kwargs):
        # Calls the classmethod on DeviceManager.
        return DeviceManager.torch_device_replacement(*args, **kwargs)

    @classmethod
    def __instancecheck__(cls, instance):
        # Uses the module-level _ORIGINAL_TORCH_DEVICE_CLASS captured at import.
        return isinstance(instance, _ORIGINAL_TORCH_DEVICE_CLASS)

_patcher_instance_for_torch_device = _TorchDevicePatcher() # Create a single instance to be used for patching

def apply_patches() -> None:
    """Apply all device-related patches."""
    log_info("Applying device patches")
    
    # Ensure DeviceManager.t_device uses the true original torch.device class.
    # _ORIGINAL_TORCH_DEVICE_CLASS was captured when the module was first imported.
    DeviceManager.t_device = _ORIGINAL_TORCH_DEVICE_CLASS
    
    # Initialize default device using original mechanisms before patching torch.device globally.
    # DeviceManager.get_default_device() calls cls.t_device(), which is now correctly set.
    DeviceManager.get_default_device() 
    
    # Patch torch.device with our callable instance that also handles isinstance correctly.
    torch.device = _patcher_instance_for_torch_device
    
    log_info("Device patches applied")


# Module initialization
log_info("Initializing TorchDevice core device module")

__all__: list[str] = [
    'DeviceManager',
    'apply_patches',
    '_ORIGINAL_TORCH_DEVICE_CLASS'
]

log_info("TorchDevice core device module initialized")
