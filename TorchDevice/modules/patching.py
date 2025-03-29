"""
Patching  functions and hooks
"""
import torch
from .TDLogger import auto_log, log_info
from .device_detection import _CACHED_DEFAULT_DEVICE

# Save original functions.
_original_torch_load = torch.load
_original_tensor_cuda = torch.Tensor.cuda
_original_module_cuda = torch.nn.Module.cuda
_original_tensor_to = torch.Tensor.to
_original_module_to = torch.nn.Module.to

@auto_log()
def tensor_cuda_replacement(self, *args, **kwargs):
    cached_device_type = _CACHED_DEFAULT_DEVICE
    if cached_device_type != 'cuda':
        return self.to(cached_device_type, **kwargs)
    return _original_tensor_cuda(self, *args, **kwargs)

@auto_log()
def module_cuda_replacement(self, *args, **kwargs):
    cached_device_type = _CACHED_DEFAULT_DEVICE
    if cached_device_type != 'cuda':
        return self.to(cached_device_type, **kwargs)
    return _original_module_cuda(self, *args, **kwargs)

@auto_log()
def tensor_to_replacement(self, *args, **kwargs):
    cached_device_type = _CACHED_DEFAULT_DEVICE
    def normalize(dev_str):
        if dev_str == 'cpu:-1':
            # CPU override: use torch.device('cpu:0') and flag override in the main class.
            from ..TorchDevice import TorchDevice
            with TorchDevice._lock:
                TorchDevice._default_device = 'cpu'
                TorchDevice._cpu_override = True
            return 'cpu:0'
        return cached_device_type
    if args and isinstance(args[0], str):
        target_device = normalize(args[0])
        new_args = (target_device,) + args[1:]
        return _original_tensor_to(self, *new_args, **kwargs)
    elif 'device' in kwargs and isinstance(kwargs['device'], str):
        kwargs['device'] = normalize(kwargs['device'])
        return _original_tensor_to(self, *args, **kwargs)
    else:
        return _original_tensor_to(self, *args, **kwargs)

@auto_log()
def module_to_replacement(self, *args, **kwargs):
    cached_device_type = _CACHED_DEFAULT_DEVICE
    def normalize(dev_str):
        if dev_str == 'cpu:-1':
            from ..TorchDevice import TorchDevice
            with TorchDevice._lock:
                TorchDevice._default_device = 'cpu'
                TorchDevice._cpu_override = True
            return 'cpu:0'
        return cached_device_type
    if args and isinstance(args[0], str):
        target_device = normalize(args[0])
        new_args = (target_device,) + args[1:]
        return _original_module_to(self, *new_args, **kwargs)
    elif 'device' in kwargs and isinstance(kwargs['device'], str):
        kwargs['device'] = normalize(kwargs['device'])
        return _original_module_to(self, *args, **kwargs)
    else:
        return _original_module_to(self, *args, **kwargs)

@auto_log()
def torch_load_replacement(*args, **kwargs):
    global _in_torch_load
    try:
        _in_torch_load
    except NameError:
        _in_torch_load = False
    if _in_torch_load:
        return _original_torch_load(*args, **kwargs)
    _in_torch_load = True
    try:
        cached_device_type = _CACHED_DEFAULT_DEVICE
        if 'map_location' in kwargs:
            if kwargs['map_location'] == 'cpu' or (isinstance(kwargs['map_location'], str) and kwargs['map_location'] != cached_device_type):
                kwargs['map_location'] = cached_device_type
        else:
            kwargs['map_location'] = cached_device_type
        return _original_torch_load(*args, **kwargs)
    finally:
        _in_torch_load = False

@auto_log()
def apply_basic_patches():
    """
    Apply basic patches for torch functions.
    """
    torch.Tensor.cuda = tensor_cuda_replacement
    torch.nn.Module.cuda = module_cuda_replacement
    torch.Tensor.to = tensor_to_replacement
    torch.nn.Module.to = module_to_replacement
    torch.load = torch_load_replacement

# --- AMP Hooks (retain original behavior) ---
if hasattr(torch.cuda, 'amp'):
    _original_autocast = torch.cuda.amp.autocast
    @auto_log()
    def autocast_replacement(*args, **kwargs):
        cached_device_type = _CACHED_DEFAULT_DEVICE
        if cached_device_type != 'cuda':
            log_info("torch.cuda.amp.autocast called on a non-CUDA device; behavior may be unexpected with autocast.")
        return _original_autocast(*args, **kwargs)
    torch.cuda.amp.autocast = autocast_replacement

    if hasattr(torch.cuda.amp, 'GradScaler'):
        _OriginalGradScaler = torch.cuda.amp.GradScaler
        class GradScalerReplacement(_OriginalGradScaler):
            @auto_log()
            def __init__(self, *args, **kwargs):
                cached_device_type = _CACHED_DEFAULT_DEVICE
                if cached_device_type != 'cuda':
                    log_info("torch.cuda.amp.GradScaler instantiated on a non-CUDA device; behavior may be unexpected with GradScaler.")
                super().__init__(*args, **kwargs)
        torch.cuda.amp.GradScaler = GradScalerReplacement

# --- Patch CUDA lazy init (prevent errors on systems without CUDA) ---
torch.cuda._lazy_init = lambda: None