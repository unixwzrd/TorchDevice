"""
TorchDevice Hardware Information Module
-------------------------------------
Provides direct, unpatched access to PyTorch hardware and compilation status.
This module should not have dependencies on other TorchDevice modules to avoid
circular imports during early initialization.
"""

import torch
from .logger import log_info

_is_pytorch_compiled_with_cuda_value = hasattr(torch._C, '_cuda_getDeviceCount')
_is_native_cuda_available_value = torch.cuda.is_available() if _is_pytorch_compiled_with_cuda_value else False
_is_native_cuda_built_value = torch.backends.cuda.is_built() if hasattr(torch.backends, 'cuda') else False
_is_native_mps_available_value = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
_is_native_mps_built_value = torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False

log_info("[HW_INFO] PyTorch compiled with CUDA: %s", _is_pytorch_compiled_with_cuda_value)
log_info("[HW_INFO] Native CUDA available: %s", _is_native_cuda_available_value)
log_info("[HW_INFO] Native CUDA built: %s", _is_native_cuda_built_value)
log_info("[HW_INFO] Native MPS available: %s", _is_native_mps_available_value)
log_info("[HW_INFO] Native MPS built: %s", _is_native_mps_built_value)

def is_pytorch_compiled_with_cuda() -> bool:
    """Returns True if PyTorch was compiled with CUDA support (native check)."""
    return _is_pytorch_compiled_with_cuda_value

def is_native_cuda_available() -> bool:
    """Returns True if CUDA is natively available and usable by PyTorch (native check)."""
    return _is_native_cuda_available_value

def is_native_cuda_built() -> bool:
    """Returns True if PyTorch's CUDA backend is built (native check)."""
    return _is_native_cuda_built_value

def is_native_mps_available() -> bool:
    """Returns True if MPS is natively available and usable by PyTorch (native check)."""
    return _is_native_mps_available_value

def is_native_mps_built() -> bool:
    """Returns True if PyTorch's MPS backend is built (native check)."""
    return _is_native_mps_built_value

__all__ = [
    'is_pytorch_compiled_with_cuda',
    'is_native_cuda_available',
    'is_native_cuda_built',
    'is_native_mps_available',
    'is_native_mps_built'
]
