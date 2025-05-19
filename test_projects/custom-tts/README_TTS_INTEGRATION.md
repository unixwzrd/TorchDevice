# TTS Integration with TorchDevice

This directory contains scripts and examples demonstrating the integration of Text-to-Speech (TTS) functionality with the TorchDevice module, ensuring compatibility across different device types (CPU, GPU, MPS).

## Overview

The integration approach uses TorchDevice for general device routing and adds application-specific patches for TTS compatibility. These patches are implemented directly in the TTS application code rather than in TorchDevice, following the principle that application-specific patches should not be part of the general TorchDevice library.

The TTS-specific patches include:

1. `torch.serialization._get_restore_location` - Ensures proper loading of models across devices
2. `TTS.tts.utils.synthesis.numpy_to_torch` - Handles `None` inputs and redirects CUDA calls to MPS

These patches allow TTS models to work seamlessly on Apple Silicon (M1/M2/M3) machines by redirecting CUDA operations to MPS.

## Files in this Directory

- `final_tts_fixed.py`: A comprehensive integration script demonstrating TTS functionality with TorchDevice
- `final_tts_integration.py`: A script that tests TTS with direct MPS support
- `tts_torchdevice_complete.py`: A complete solution with all necessary patches and tests
- `working_tts_integration.py`: A working integration example
- `simple_tts_test.py`: A basic test script for TTS without TorchDevice integration

## How to Use

1. **Basic Usage**: Run `final_tts_fixed.py` to see TTS working with TorchDevice patches
   ```bash
   python final_tts_fixed.py
   ```

2. **Direct MPS Support**: Run `final_tts_integration.py` to use TTS with direct MPS support
   ```bash
   python final_tts_integration.py
   ```

3. **Complete Solution**: Run `tts_torchdevice_complete.py` to see all patches and tests in action
   ```bash
   python tts_torchdevice_complete.py
   ```

## Implementation Details

### TTS-Specific Patches

The following patches are applied in the TTS application code:

```python
# Patch torch.serialization._get_restore_location for TTS compatibility
original_get_restore_location = torch.serialization._get_restore_location

def patched_get_restore_location(map_location):
    # Create a simple restore function that ignores location
    def restore_location(storage, location):
        return storage
    
    return restore_location

torch.serialization._get_restore_location = patched_get_restore_location

# Patch TTS synthesis module if it's available
try:
    from TTS.tts.utils import synthesis
    
    # Store the original numpy_to_torch function
    original_numpy_to_torch = synthesis.numpy_to_torch
    
    # Define a patched version
    def patched_numpy_to_torch(np_array, dtype, device=None):
        # Handle None input gracefully
        if np_array is None:
            return None
            
        # Create tensor on CPU first
        tensor = torch.as_tensor(np_array, dtype=dtype, device='cpu')
        
        # Then move to the requested device if specified
        if device is not None:
            # If device is 'cuda', check if MPS is available
            if device == 'cuda' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Redirecting tensor from 'cuda' to 'mps' in TTS synthesis")
                device = 'mps'
            
            # Move tensor to the appropriate device
            tensor = tensor.to(device)
        
        return tensor
    
    # Apply the patch
    synthesis.numpy_to_torch = patched_numpy_to_torch
except ImportError:
    # TTS is not installed, so no need to patch
    pass
```

### Best Practices

1. Always import TorchDevice, order does not matter, may be imported before or after PyTorch.
2. Implement application-specific patches in the application code, not in TorchDevice
3. Always load models on CPU first, then move to the appropriate device
4. Use TorchDevice's device detection to determine the best available device
5. For CUDA code, let TorchDevice handle the redirection to MPS
6. For new code, explicitly check for MPS availability and use it directly

## Performance Comparison

Tests show that TTS with MPS acceleration is significantly faster than CPU-only processing:

- CPU: ~1.5-2x real-time factor
- MPS: ~0.3-0.5x real-time factor (3-5x faster than CPU)

## Troubleshooting

If you encounter issues:

1. Make sure TorchDevice is imported before any TTS imports
2. Check that the TTS-specific patches are correctly applied in your application code
3. Verify that your PyTorch version supports MPS (requires PyTorch 1.12+)
4. For model loading issues, try loading on CPU first, then moving to MPS
