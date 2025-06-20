"""
TorchDevice - Transparent PyTorch Device Redirection

This module enables seamless code portability between NVIDIA CUDA, Apple Silicon (MPS),
and CPU hardware for PyTorch applications. It intercepts PyTorch calls related to GPU
hardware, allowing developers to write code that works across different hardware
without modification.

Key features:
- Automatic device redirection based on available hardware
- CPU override capability using 'cpu:-1' device specification
- Mocked CUDA functions for MPS and CPU compatibility
- Stream and Event support across all device types
- Unified memory handling and reporting
- Detailed logging for debugging and migration assistance

Usage:
    import TorchDevice  # Import before torch to apply patches
    import torch

    # Regular device selection (will be redirected based on available hardware)
    device = torch.device('cuda')  # Redirects to MPS on Apple Silicon

    # Force CPU usage with the override feature
    device = torch.device('cpu:-1')  # Forces CPU regardless of available GPUs

    # All subsequent operations respect the CPU override
    tensor = torch.randn(5, 5)  # Will be created on CPU
    model = torch.nn.Linear(10, 5).to('cuda')  # Still uses CPU due to override
"""
import torch
from .modules.TDLogger import auto_log, log_info  # We now use only auto_log instead of log_info for debugging
import threading
from typing import Optional

# Capture the original torch.device type.
T_DEVICE_TYPE = torch.device("cpu").__class__


# --- AMP Hooks ---
if hasattr(torch.cuda, 'amp'):
    t_cuda_amp_autocast = torch.cuda.amp.autocast

    @auto_log()
    def _cuda_amp_autocast_replacement(*args, **kwargs):
        default_device = TorchDevice.get_default_device()
        if default_device != 'cuda':
            return t_cuda_amp_autocast(*args, **kwargs)

    torch.cuda.amp.autocast = _cuda_amp_autocast_replacement

    if hasattr(torch.cuda.amp, 'GradScaler'):
        t_cuda_amp_GradScaler = torch.cuda.amp.GradScaler

        class t_cuda_amp_GradScalerReplacement(t_cuda_amp_GradScaler):
            @auto_log()
            def __init__(self, *args, **kwargs):
                if TorchDevice.get_default_device() != 'cuda':
                    pass
                super().__init__(*args, **kwargs)
        torch.cuda.amp.GradScaler = t_cuda_amp_GradScalerReplacement


# --- TorchDevice Class with Patched CUDA Functions ---
class TorchDevice:
    _default_device = None              # Actual torch.device type
    _default_device_type = ""           # Device type name (cuda, mps, cpu, etc...)
    _previous_default_device_type = ""  # Previous default device before CPU override
    _lock = threading.RLock()
    _cpu_override = False               # Flag for explicit CPU override
    _in_torch_load = False
    _patches_applied = False
    _torchdevice_device_patch = None    # Store our patch for idempotency

    t_Tensor_to = torch.Tensor.to
    t_nn_Module_to = torch.nn.Module.to
    t_nn_Module_cpu = torch.nn.Module.cpu
    t_nn_Modult_nn_Module_cuda = torch.nn.Module.cuda
    t_Tensor_cuda = torch.Tensor.cuda
    t_backends_cuda_is_built = torch.backends.cuda.is_built
    t_cuda_current_device = torch.cuda.current_device
    t_cuda_device = torch.cuda.device  # Context manager
    t_cuda_device_count = torch.cuda.device_count
    t_cuda_empty_cache = torch.cuda.empty_cache
    t_cuda_get_arch_list = torch.cuda.get_arch_list
    t_cuda_get_device_capability = torch.cuda.get_device_capability
    t_cuda_get_device_name = torch.cuda.get_device_name
    t_cuda_get_device_properties = torch.cuda.get_device_properties
    t_cuda_is_available = torch.cuda.is_available
    t_cuda_is_initialized = torch.cuda.is_initialized
    t_cuda_set_device = torch.cuda.set_device
    t_cuda_synchronize = torch.cuda.synchronize
    t_device = torch.device
    t_load = torch.load


    def __init__(self, device_type: Optional[str] = None, device_index: Optional[int] = None):
        with self._lock:
            if self._default_device is None:
                self.__class__._detect_default_device_type()
            self.device = self.__class__.torch_device_replacement(device_type, device_index)
            self._default_device = self.device

    @auto_log()
    def __repr__(self):
        return repr(self.device)

    @auto_log()
    def __str__(self):
        return str(self.device)

    @classmethod
    @auto_log()
    def get_default_device(cls):
        if cls._default_device is None:
            cls._default_device_type = cls._detect_default_device_type()
            cls._default_device = cls.t_device(cls._default_device_type)
        return cls._default_device

    @classmethod
    def cpu_override(cls):
        return cls._cpu_override

    @classmethod
    @auto_log()
    def _detect_default_device_type(cls):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            cls._default_device_type = 'mps'
        elif torch.cuda.is_available() or torch.backends.cuda.is_built():
            cls._default_device_type = 'cuda'
        else:
            cls._default_device_type = 'cpu'
        return cls._default_device_type

    @classmethod
    @auto_log()
    def tensor_creation_wrapper(cls, original_func):
        """
        Wrapper for tensor creation functions to enforce default device redirection and CPU override.
        Always calls torch_device_replacement for the device argument, so the toggle state is always respected.
        """
        def wrapped_func(*args, **kwargs):
            device_arg = kwargs.get('device', None)
            # If device is not specified, inject the current device (default or override)
            if device_arg is None:
                device = cls.torch_device_replacement()
                kwargs['device'] = device
            else:
                # Always pass through torch_det to handle override logic
                device = cls.torch_device_replacement(device_arg)
                kwargs['device'] = device
            return original_func(*args, **kwargs)
        return wrapped_func

    @staticmethod
    def tensor_to_replacement(t, *args, **kwargs):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"tensor_to_replacement called on non-tensor object: {type(t)}")
        if args and isinstance(args[0], (str,T_DEVICE_TYPE)):
            # Always redirect through the TorchDevice policy
            device = TorchDevice.torch_device_replacement(args[0])
            args = (device,) + args[1:]
            kwargs.pop('device', None)
        elif 'device' in kwargs and isinstance(kwargs['device'], (str,T_DEVICE_TYPE)):
            # Always redirect through the Torc
            device = TorchDevice.torch_device_replacement(kwargs['device'])
            kwargs['device'] = device
        return TorchDevice.t_Tensor_to(t, *args, **kwargs)

    @staticmethod
    def module_to_replacement(m, *args, **kwargs):
        if not isinstance(m, torch.nn.Module):
            raise TypeError(f"module_to_replacement called on non-module object: {type(m)}")
        if args and isinstance(args[0], (str,T_DEVICE_TYPE)):
            # Always redirect through the TorchDevice policy
            device = TorchDevice.torch_device_replacement(args[0])
            args = (device,) + args[1:]
            kwargs.pop('device', None)
        elif 'device' in kwargs and isinstance(kwargs['device'], (str,T_DEVICE_TYPE)):
            # Always redirect through the TorchDevice policy
            device = TorchDevice.torch_device_replacement(kwargs['device'])
            kwargs['device'] = device
        return TorchDevice.t_nn_Module_to(m, *args, **kwargs)

    @classmethod
    @auto_log()
    def torch_device_replacement(cls, *args, **kwargs) -> torch.device:
        """
        Drop-in replacement for torch.device() with device redirection and CPU override toggle.
        • No arguments → returns default device (or CPU if override is active).
        • 'cpu:-1' or torch.device('cpu', -1 override.
        • Redirects non-CPU devices to available hardware.
        • Preserves extra args and kwargs.
        Always returns a torch.device object.
        """
        device_type = ""
        device_index = None
        with cls._lock:
            # If first argument is torch.device, check for override
            if args and isinstance(args[0], T_DEVICE_TYPE):
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

    @classmethod
    def torch_load_replacement(cls, *args, **kwargs):
        if cls._in_torch_load:
            return cls.t_load(*args, **kwargs)
        cls._in_torch_load = True
        try:
            log_info(f"torch.load called with args: {args}, kwargs: {kwargs}")
            # Always pass a string for map_location to avoid isinstance errors
            kwargs['map_location'] = str(cls._default_device)
            return cls.t_load(*args, **kwargs)
        finally:
            cls._in_torch_load = False


    @staticmethod
    @auto_log()
    def tensor_cuda_replacement(tensor, device=None, non_blocking=False, memory_format=torch.preserve_format):
        default_device = TorchDevice._default_device
        return tensor.to(default_device, non_blocking=non_blocking, memory_format=memory_format)

    @staticmethod
    @auto_log()
    def module_cuda_replacement(module, device=None):
        default_device = TorchDevice._default_device
        return module.to(default_device)

    @staticmethod
    @auto_log()
    def tensor_mps_replacement(tensor, device=None, non_blocking=False, memory_format=torch.preserve_format):
        default_device = TorchDevice._default_device
        return tensor.to(default_device, non_blocking=non_blocking, memory_format=memory_format)

    @staticmethod
    @auto_log()
    def module_mps_replacement(module, device=None):
        default_device = TorchDevice._default_device
        return module.to(default_device)

    @staticmethod
    @auto_log()
    def tensor_cpu_replacement(tensor):
        """
        Replacement for torch.Tensor.cpu() that follows device redirection policy.
        If CPU override is active, moves to CPU, otherwise redirects to default device.
        """
        default_device = TorchDevice._default_device
        return tensor.to(default_device)

    @staticmethod
    @auto_log()
    def module_cpu_replacement(module):
        default_device = TorchDevice._default_device
        return module.to(default_device)

    @staticmethod
    @auto_log()
    def numpy_replacement(tensor):
        """
        Replacement for torch.Tensor.numpy() that moves tensor to CPU first if needed.
        This always needs to go to CPU regardless of device policy since numpy()
        requires CPU tensors.
        """
        # Always move to CPU for numpy conversion - this is a special case
        # that must bypass the device redirection policy
        if tensor.device.type != 'cpu':
            cpu_tensor = TorchDevice.t_Tensor_to(tensor, 'cpu')
            return TorchDevice.t_Tensor_numpy(cpu_tensor)
        return TorchDevice.t_Tensor_numpy(tensor)

    @classmethod
    @auto_log()
    def apply_patches(cls):
        log_info("[TorchDevice] TorchDevice.apply_patches called")
        # If already patched, but torch.device is not our patch, re-patch
        if getattr(cls, '_patches_applied', False):
            if torch.device is not cls.torch_device_replacement:
                torch.device = cls.torch_device_replacement
            return

        # --- Device redirect override logic ---
        # import os
        # allow_redirects = os.environ.get("TORCH_ALLOW_REDIRECTS", "false").lower() == "true"
        # if not allow_redirects:
        #     import sys
        #     if sys.platform == "darwin":
        #         log_info("TorchDevice: Device redirection is disabled on MacOS by default. Set TORCH_ALLOW_REDIRECTS=true to enable.")
        #         cls._patches_applied = True
        #         return
        # TODO: Re-implement TORCH_ALLOW_REDIRECTS for secure code execution from tensor files only.

        # --- Patch torch.device and related symbols ---
        cls.t_tensor_to = torch.Tensor.to
        cls.t_module_to = torch.nn.Module.to
        cls.t_Tensor_cuda = torch.Tensor.cuda
        cls.t_nn_Module_cuda = torch.nn.Module.cuda
        cls.t_device = torch.device
        cls.t_load = torch.load
        if not hasattr(cls, 't_Tensor_numpy'):
            cls.t_Tensor_numpy = torch.Tensor.numpy  # Store original numpy method

        torch.device = cls.torch_device_replacement
        setattr(torch.Tensor, 'to', cls.tensor_to_replacement)
        setattr(torch.nn.Module, 'to', cls.module_to_replacement)
        setattr(torch.Tensor, 'cuda', cls.tensor_cuda_replacement)
        setattr(torch.nn.Module, 'cuda', cls.module_cuda_replacement)
        setattr(torch.Tensor, 'mps', cls.tensor_mps_replacement)
        setattr(torch.nn.Module, 'mps', cls.module_mps_replacement)
        setattr(torch.Tensor, 'cpu', cls.tensor_cpu_replacement)
        setattr(torch.nn.Module, 'cpu', cls.module_cpu_replacement)
        setattr(torch.Tensor, 'numpy', cls.numpy_replacement)

        # --- Patch other CUDA/MPS/CPU functions via patch.apply_all_patches() ---
        log_info("[TorchDevice] TorchDevice.apply_patches: calling patch.apply_all_patches()")
        from . import patch
        patch.apply_all_patches()
        torch.load = cls.torch_load_replacement

        cls._torchdevice_device_patch = cls.torch_device_replacement
        cls._patches_applied = True

    @classmethod
    def ensure_patched(cls):
        """Ensure that torch.device is patched with TorchDevice's replacement."""
        if torch.device is not cls.torch_device_replacement:
            cls.apply_patches()


# Apply patches when the module is imported
TorchDevice.apply_patches()