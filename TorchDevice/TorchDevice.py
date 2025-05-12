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
import TorchDevice.modules.patch as cuda_patch
import threading
from typing import Optional

# Capture the original torch.device type.
_ORIGINAL_TORCH_DEVICE_TYPE = torch.device("cpu").__class__

_CACHED_DEFAULT_DEVICE = None
_device_type = None

# --- AMP Hooks ---
if hasattr(torch.cuda, 'amp'):
    _original_autocast = torch.cuda.amp.autocast

    @auto_log()
    def autocast_replacement(*args, **kwargs):
        default_device = TorchDevice.get_default_device()
        if default_device != 'cuda':
            return _original_autocast(*args, **kwargs)

    torch.cuda.amp.autocast = autocast_replacement

    if hasattr(torch.cuda.amp, 'GradScaler'):
        _OriginalGradScaler = torch.cuda.amp.GradScaler

        class GradScalerReplacement(_OriginalGradScaler):
            @auto_log()
            def __init__(self, *args, **kwargs):
                if TorchDevice.get_default_device() != 'cuda':
                    pass
                super().__init__(*args, **kwargs)
        torch.cuda.amp.GradScaler = GradScalerReplacement


# --- TorchDevice Class with Patched CUDA Functions ---
class TorchDevice:
    _default_device = None
    _previous_default_device = None
    _lock = threading.RLock()
    _cpu_override = False  # Flag for explicit CPU override
    _in_torch_load = False
    _patches_applied = False

    _original_tensor_to = torch.Tensor.to
    _original_module_to = torch.nn.Module.to
    _original_module_cpu = torch.nn.Module.cpu
    _original_module_cuda = torch.nn.Module.cuda
    _original_tensor_cuda = torch.Tensor.cuda
    _original_torch_backends_cuda_is_built = torch.backends.cuda.is_built
    _original_torch_cuda_current_device = torch.cuda.current_device
    _original_torch_cuda_device = torch.cuda.device  # Context manager
    _original_torch_cuda_device_count = torch.cuda.device_count
    _original_torch_cuda_empty_cache = torch.cuda.empty_cache
    _original_torch_cuda_get_arch_list = torch.cuda.get_arch_list
    _original_torch_cuda_get_device_capability = torch.cuda.get_device_capability
    _original_torch_cuda_get_device_name = torch.cuda.get_device_name
    _original_torch_cuda_get_device_properties = torch.cuda.get_device_properties
    _original_torch_cuda_is_available = torch.cuda.is_available
    _original_torch_cuda_is_initialized = torch.cuda.is_initialized
    _original_torch_cuda_set_device = torch.cuda.set_device
    _original_torch_cuda_synchronize = torch.cuda.synchronize
    _original_torch_device = torch.device
    _original_torch_load = torch.load

    @auto_log()
    def __init__(self, device_type: Optional[str] = None, device_index: int = 0):
        with self._lock:
            if self._default_device is None:
                self.__class__._detect_default_device()
            if isinstance(device_type, str):
                if ':' in device_type:
                    device_type, index = device_type.split(':')
                    device_index = int(index)
                else:
                    device_index = 0 if device_index is None else device_index
                device_type = self.__class__.redirect_device_type(device_type)
                device_str = f"{device_type}:{device_index}"
                self.device = self.__class__._original_torch_device(device_str)

    @auto_log()
    def __repr__(self):
        return repr(self.device)

    @auto_log()
    def __str__(self):
        return str(self.device)

    class TorchDeviceWrapper(object):
        @auto_log()
        def __init__(self, device):
            self._device = device
            
        def __getattr__(self, name):
            return getattr(self._device, name)
            
        def __repr__(self):
            return repr(self._device)
            
        def __str__(self):
            return str(self._device)
            
        def __instancecheck__(self, instance):
            return isinstance(instance, _ORIGINAL_TORCH_DEVICE_TYPE)
    
    @classmethod
    @auto_log()
    def get_default_device(cls):
        """
        Return the default device based on available hardware and cache the result.
        """
        global _CACHED_DEFAULT_DEVICE
        if _CACHED_DEFAULT_DEVICE is None:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                _CACHED_DEFAULT_DEVICE = 'mps'
            elif cls._original_torch_cuda_is_available():
                _CACHED_DEFAULT_DEVICE = 'cuda'
            else:
                _CACHED_DEFAULT_DEVICE = 'cpu'
        return _CACHED_DEFAULT_DEVICE

    @classmethod
    def cpu_override_set(cls):
        return cls._cpu_override

    @classmethod
    @auto_log()
    def redirect_device_type(cls, device_type):
        """
        Redirect a device type string based on availability and CPU override.
        If cpu_override is True, always returns 'cpu'.
        For 'cuda' and 'mps' requests, return the type that is available.
        """
        # For explicit CPU requests, always return 'cpu'
        if device_type == 'cpu':
            return 'cpu'
        
        if device_type.startswith('cuda'):
            if _CACHED_DEFAULT_DEVICE == 'cuda':
                device_type = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_type = 'mps'
            else:
                device_type = 'cpu'
        elif device_type.startswith('mps'):
            if _CACHED_DEFAULT_DEVICE == 'mps':
                device_type = 'mps'
            elif cls._original_torch_cuda_is_available():
                device_type = 'cuda'
            else:
                device_type = 'cpu'
        return device_type

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
                log_info(f"[tensor_creation_wrapper] Injecting device: {device}")
                kwargs['device'] = device
            else:
                # Always pass through torch_device_replacement to handle override logic
                device = cls.torch_device_replacement(device_arg)
                log_info(f"[tensor_creation_wrapper] Normalized device: {device}")
                kwargs['device'] = device
            return original_func(*args, **kwargs)
        return wrapped_func

    @staticmethod
    def tensor_to_replacement(t, *args, **kwargs):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"tensor_to_replacement called on non-tensor object: {type(t)}")
        if args and isinstance(args[0], (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
            # Always redirect through the TorchDevice policy
            device = TorchDevice.torch_device_replacement(args[0])
            new_args = (device,) + args[1:]
            kwargs.pop('device', None)
            return TorchDevice._original_tensor_to(t, *new_args, **kwargs)
        elif 'device' in kwargs and isinstance(kwargs['device'], (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
            # Always redirect through the TorchDevice policy
            device = TorchDevice.torch_device_replacement(kwargs['device'])
            kwargs['device'] = device
            return TorchDevice._original_tensor_to(t, *args, **kwargs)
        else:
            return TorchDevice._original_tensor_to(t, *args, **kwargs)

    @staticmethod
    def module_to_replacement(m, *args, **kwargs):
        if not isinstance(m, torch.nn.Module):
            raise TypeError(f"module_to_replacement called on non-module object: {type(m)}")
        if args and isinstance(args[0], (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
            # Always redirect through the TorchDevice policy
            device = TorchDevice.torch_device_replacement(args[0])
            new_args = (device,) + args[1:]
            kwargs.pop('device', None)
            return TorchDevice._original_module_to(m, *new_args, **kwargs)
        elif 'device' in kwargs and isinstance(kwargs['device'], (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
            # Always redirect through the TorchDevice policy
            device = TorchDevice.torch_device_replacement(kwargs['device'])
            kwargs['device'] = device
            return TorchDevice._original_module_to(m, *args, **kwargs)
        else:
            return TorchDevice._original_module_to(m, *args, **kwargs)

    @classmethod
    @auto_log()
    def torch_device_replacement(cls, *args, **kwargs) -> torch.device:
        """
        Drop-in replacement for torch.device() with device redirection and CPU override toggle.
        • No arguments → returns default device (or CPU if override is active).
        • 'cpu:-1' or torch.device('cpu', -1) → toggles CPU override.
        • Redirects non-CPU devices to available hardware.
        • Preserves extra args and kwargs.
        Always returns a torch.device object.
        """
        global _CACHED_DEFAULT_DEVICE
        device_type = ""
        device_index = None
        log_info(f"Called with args={args}, kwargs={kwargs}")
        with cls._lock:
            # If first argument is torch.device, check for override
            if args and isinstance(args[0], _ORIGINAL_TORCH_DEVICE_TYPE):
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
                        if cls.cpu_override_set():
                            # Toggle OFF
                            cls._cpu_override = False
                            _CACHED_DEFAULT_DEVICE = cls._previous_default_device
                            cls._previous_default_device = None
                            log_info("CPU override toggled OFF")
                        else:
                            # Toggle ON
                            cls._cpu_override = True
                            cls._previous_default_device = _CACHED_DEFAULT_DEVICE
                            _CACHED_DEFAULT_DEVICE = 'cpu'
                            log_info("CPU override toggled ON")
                
            device_type = _CACHED_DEFAULT_DEVICE
            result = cls._original_torch_device(device_type, device_index)
            return result
    
    @classmethod
    @auto_log()
    def torch_load_replacement(cls, *args, **kwargs):
        if cls._in_torch_load:
            return cls._original_torch_load(*args, **kwargs)
        cls._in_torch_load = True
        try:
            default_device = cls.get_default_device()
            if 'map_location' in kwargs:
                if kwargs['map_location'] == 'cpu' or (
                    isinstance(kwargs['map_location'], str) and kwargs['map_location'] != default_device
                ):
                    kwargs['map_location'] = default_device
            else:
                kwargs['map_location'] = default_device
            return cls._original_torch_load(*args, **kwargs)
        finally:
            cls._in_torch_load = False

    @classmethod
    @auto_log()
    def _detect_default_device(cls):
        if torch.backends.mps.is_available():
            cls._default_device = 'mps'
        elif cls._original_torch_cuda_is_available():
            cls._default_device = 'cuda'
        else:
            cls._default_device = 'cpu'

    @staticmethod
    @auto_log()
    def tensor_cuda_replacement(tensor, device=None, non_blocking=False, memory_format=torch.preserve_format):
        default_device = TorchDevice.get_default_device()
        return tensor.to(default_device, non_blocking=non_blocking, memory_format=memory_format)

    @staticmethod
    @auto_log()
    def module_cuda_replacement(module, device=None):
        default_device = TorchDevice.get_default_device()
        return module.to(default_device)

    @staticmethod
    @auto_log()
    def tensor_mps_replacement(tensor, device=None, non_blocking=False, memory_format=torch.preserve_format):
        default_device = TorchDevice.get_default_device()
        return tensor.to(default_device, non_blocking=non_blocking, memory_format=memory_format)

    @staticmethod
    @auto_log()
    def module_mps_replacement(module, device=None):
        default_device = TorchDevice.get_default_device()
        return module.to(default_device)

    @staticmethod
    @auto_log()
    def tensor_cpu_replacement(tensor):
        """
        Replacement for torch.Tensor.cpu() that follows device redirection policy.
        If CPU override is active, moves to CPU, otherwise redirects to default device.
        """
        # If CPU override is active, actually use CPU
        if TorchDevice.cpu_override_set():
            return TorchDevice._original_tensor_to(tensor, 'cpu')
        # Otherwise redirect to default device as per policy
        default_device = TorchDevice.get_default_device()
        return tensor.to(default_device)

    @staticmethod
    @auto_log()
    def module_cpu_replacement(module):
        default_device = TorchDevice.get_default_device()
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
            cpu_tensor = TorchDevice._original_tensor_to(tensor, 'cpu')
            return TorchDevice._original_numpy(cpu_tensor)
        return TorchDevice._original_numpy(tensor)

    @classmethod
    @auto_log()
    def apply_patches(cls):
        cls.get_default_device()
        """Apply all patches to PyTorch."""
        if cls._patches_applied:
            return

        # --- Original Method Storage ---
        # Store references to original methods before patching
        cls._original_tensor_to = torch.Tensor.to
        cls._original_module_to = torch.nn.Module.to
        cls._original_tensor_cuda = torch.Tensor.cuda
        cls._original_module_cuda = torch.nn.Module.cuda
        cls._original_torch_device = torch.device
        cls._original_torch_load = torch.load
        cls._original_numpy = torch.Tensor.numpy  # Store original numpy method

        # --- Patch PyTorch Methods ---
        torch.device = cls.torch_device_replacement
        setattr(torch.Tensor, 'to', cls.tensor_to_replacement)  # type: ignore[attr-defined]
        setattr(torch.nn.Module, 'to', cls.module_to_replacement)  # type: ignore[attr-defined]
        setattr(torch.Tensor, 'cuda', cls.tensor_cuda_replacement)  # type: ignore[attr-defined]
        setattr(torch.nn.Module, 'cuda', cls.module_cuda_replacement)  # type: ignore[attr-defined]
        setattr(torch.Tensor, 'mps', cls.tensor_mps_replacement)  # type: ignore[attr-defined]
        setattr(torch.nn.Module, 'mps', cls.module_mps_replacement)  # type: ignore[attr-defined]
        setattr(torch.Tensor, 'cpu', cls.tensor_cpu_replacement)  # type: ignore[attr-defined]
        setattr(torch.nn.Module, 'cpu', cls.module_cpu_replacement)  # type: ignore[attr-defined]
        setattr(torch.Tensor, 'numpy', cls.numpy_replacement)  # type: ignore[attr-defined]
        cuda_patch.apply_all_patches()
        torch.load = cls.torch_load_replacement  # type: ignore[assignment]

        cls._patches_applied = True


# Apply patches when the module is imported
TorchDevice.apply_patches()
