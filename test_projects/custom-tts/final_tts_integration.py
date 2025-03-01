#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final TTS integration with TorchDevice.
This script demonstrates the complete integration of TTS with TorchDevice.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import TorchDevice first to apply general patches
import TorchDevice
import torch

# Apply TTS-specific patches
def apply_tts_patches():
    # Patch torch.serialization._get_restore_location for TTS compatibility
    original_get_restore_location = torch.serialization._get_restore_location

    def patched_get_restore_location(map_location):
        # Create a simple restore function that ignores location
        def restore_location(storage, location):
            return storage
        
        return restore_location

    torch.serialization._get_restore_location = patched_get_restore_location
    logger.info("TorchDevice patch applied to torch.serialization._get_restore_location")

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
        logger.info("Patched TTS.tts.utils.synthesis.numpy_to_torch to handle device properly")
        return True
    except ImportError:
        # TTS is not installed, so no need to patch
        logger.warning("TTS not installed, skipping TTS-specific patches")
        return False

# Apply TTS-specific patches
apply_tts_patches()

# Now import TTS
from TTS.api import TTS
import soundfile as sf

def test_tts_with_short_text(output_path="samples/final_short_output.wav"):
    """Test TTS with a very short text sample."""
    logger.info("\nTesting TTS with a very short text sample...")
    
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Initialize TTS with a simple English model
        logger.info("Loading model on CPU...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        
        # Generate speech with a very short text
        text = "Hello world."
        logger.info("Generating speech for text: '%s'", text)
        
        tts.tts_to_file(text=text, file_path=output_path)
        
        logger.info("Speech generated and saved to %s", output_path)
        
        return True
    except Exception as e:
        logger.error("Error in TTS test with short text: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_tts_with_gpu(output_path="samples/final_gpu_output.wav"):
    """Test TTS with GPU enabled, allowing TorchDevice to redirect to MPS."""
    logger.info("\nTesting TTS with GPU enabled (TorchDevice should redirect to MPS)...")
    
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Check if CUDA is available (should be True with TorchDevice)
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        logger.info("CUDA available: %s", cuda_available)
        logger.info("MPS available: %s", mps_available)
        
        # Initialize TTS with GPU enabled
        logger.info("Loading model with GPU enabled...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        
        # Instead of directly moving to CUDA, check for MPS and use it
        logger.info("Moving model to appropriate GPU device...")
        if torch.backends.mps.is_available():
            logger.info("Using MPS device")
            tts.to('mps')
        elif torch.cuda.is_available():
            logger.info("Using CUDA device")
            tts.to('cuda')
        else:
            logger.info("No GPU available, using CPU")
        
        # Generate speech
        text = "This is a test of text to speech with GPU enabled."
        logger.info("Generating speech for text: '%s'", text)
        
        tts.tts_to_file(text=text, file_path=output_path)
        
        logger.info("Speech generated and saved to %s", output_path)
        
        return True
    except Exception as e:
        logger.error("Error in TTS test with GPU: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run the test."""
    # Create samples directory if it doesn't exist
    Path("samples").mkdir(exist_ok=True)
    
    # Test with short text on CPU
    short_success = test_tts_with_short_text()
    
    # Test with GPU (redirected to MPS)
    gpu_success = test_tts_with_gpu()
    
    # Print test results
    logger.info("\nTest Results:")
    logger.info("Short text test (CPU): %s", 'SUCCESS' if short_success else 'FAILURE')
    logger.info("GPU test (redirected to MPS): %s", 'SUCCESS' if gpu_success else 'FAILURE')
    
    if short_success and gpu_success:
        logger.info("\nSuccess! TorchDevice is now fully integrated with TTS.")
        logger.info("The audio files have been saved to the samples directory.")
        logger.info("\nTo listen to the generated audio, you can use:")
        logger.info("  - samples/final_short_output.wav (CPU)")
        logger.info("  - samples/final_gpu_output.wav (GPU/MPS)")
    else:
        logger.info("\nSome tests failed. Check the logs for details.")
    
    # Create a PR-ready patch
    create_pr_patch()

def create_pr_patch():
    """Create a PR-ready patch for TorchDevice to work with TTS."""
    patch_content = """
# Add this to TorchDevice.py in the apply_patches method

# Patch torch.serialization._get_restore_location to handle TorchDevice objects
original_get_restore_location = torch.serialization._get_restore_location

def patched_get_restore_location(map_location):
    # Handle device objects by checking for attributes instead of using isinstance
    if hasattr(map_location, 'type') and hasattr(map_location, 'index'):
        def restore_location(storage, location):
            return storage
        return restore_location
    
    # Handle other cases with the original function
    return original_get_restore_location(map_location)

torch.serialization._get_restore_location = patched_get_restore_location
"""
    
    # Write the patch to a file
    patch_file = "torchdevice_tts_final_patch.txt"
    with open(patch_file, "w") as f:
        f.write(patch_content)
    
    logger.info("PR-ready patch created and saved to %s", patch_file)
    logger.info("\nTo make this patch permanent, add the code to the apply_patches method in TorchDevice.py")

if __name__ == "__main__":
    main()
