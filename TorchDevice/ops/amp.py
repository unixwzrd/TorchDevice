"""
TorchDevice Automatic Mixed Precision (AMP) Module
-------------------------------------------------
Handles patching and redirection for torch.cuda.amp functionalities.
"""

import torch
from TorchDevice.core.logger import log_info, auto_log
from TorchDevice.core.device import DeviceManager

_original_cuda_amp_autocast = None
_original_cuda_amp_GradScaler = None
_patches_applied = False

@auto_log()
def _autocast_replacement(*args, **kwargs):
    """Replaces torch.cuda.amp.autocast to use generic torch.autocast with proper device_type."""
    current_device_type = DeviceManager.get_default_device().type
    # log_info(f"[AMP] autocast_replacement called. Detected device_type: {current_device_type}")
    if not _original_cuda_amp_autocast: # Should not happen if patches applied correctly
        log_info("[AMP] Original torch.cuda.amp.autocast not captured. Using torch.autocast directly.")
        # Fallback, though ideally _original_cuda_amp_autocast should be the one from torch.cuda.amp
        return torch.autocast(device_type=current_device_type, *args, **kwargs)

    # If current_device_type is 'cuda', using original torch.cuda.amp.autocast is fine.
    # For 'mps' or 'cpu', torch.autocast handles it.
    return torch.autocast(device_type=current_device_type, *args, **kwargs)

class GradScalerReplacement(torch.amp.GradScaler):
    """
    Replaces torch.cuda.amp.GradScaler.
    Uses torch.amp.GradScaler and disables it if the effective device is not CUDA.
    """
    @auto_log()
    def __init__(self, *args, **kwargs):
        current_device_type = DeviceManager.get_default_device().type
        # log_info(f"[AMP] GradScalerReplacement.__init__ called. Detected device_type: {current_device_type}")
        
        effective_kwargs = kwargs.copy()
        if current_device_type != 'cuda':
            log_info("[AMP] Non-CUDA device ('%s') detected. Disabling GradScaler.", current_device_type)
            effective_kwargs['enabled'] = False
            # Set device to CPU to avoid issues if GradScaler tries to use CUDA context by default
            if 'device' not in effective_kwargs:
                 effective_kwargs['device'] = 'cpu' 
        elif 'device' not in effective_kwargs: # CUDA device, ensure device is set if not provided
            effective_kwargs['device'] = 'cuda'
            
        super().__init__(*args, **effective_kwargs)
        # log_info(f"[AMP] GradScalerReplacement initialized with enabled={self.is_enabled()}, scale={self.get_scale() if self.is_enabled() else 'N/A'}")

@auto_log()
def apply_patches() -> None:
    """Apply AMP related patches."""
    global _original_cuda_amp_autocast, _original_cuda_amp_GradScaler, _patches_applied
    if _patches_applied:
        return

    log_info("Applying AMP patches")
    if hasattr(torch.cuda, 'amp'):
        if hasattr(torch.cuda.amp, 'autocast'):
            _original_cuda_amp_autocast = torch.cuda.amp.autocast
            torch.cuda.amp.autocast = _autocast_replacement
            log_info("Patched torch.cuda.amp.autocast")
        else:
            log_info("torch.cuda.amp.autocast not found, skipping patch")

        if hasattr(torch.cuda.amp, 'GradScaler'):
            _original_cuda_amp_GradScaler = torch.cuda.amp.GradScaler
            torch.cuda.amp.GradScaler = GradScalerReplacement
            log_info("Patched torch.cuda.amp.GradScaler")
        else:
            log_info("torch.cuda.amp.GradScaler not found, skipping patch")
        _patches_applied = True
    else:
        log_info("torch.cuda.amp module not found, skipping AMP patches")
    log_info("AMP patches applied")

__all__ = ['GradScalerReplacement', 'apply_patches']

log_info("TorchDevice AMP module initialized")
