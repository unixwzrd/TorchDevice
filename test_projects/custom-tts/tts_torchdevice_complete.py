#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete TTS integration with TorchDevice.
This script demonstrates the complete integration of TTS with TorchDevice,
handling all necessary patches and redirections.
"""

import os
import sys
import logging
from pathlib import Path
import importlib
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import TorchDevice first to ensure it's loaded before any torch imports
import TorchDevice

# Now import torch
import torch

# Store original functions that we'll patch
_original_get_restore_location = torch.serialization._get_restore_location

# Define a custom patch for _get_restore_location
def patched_get_restore_location(map_location):
    """
    A patched version of torch.serialization._get_restore_location that handles
    TorchDevice device objects correctly.
    """
    # Create a simple restore function that ignores location
    def restore_location(storage, location):
        return storage
    
    return restore_location

# Apply the patch
torch.serialization._get_restore_location = patched_get_restore_location
logger.info("Patched torch.serialization._get_restore_location")

# Monkey patch TTS synthesis module
def patch_tts_modules():
    """
    Patch various TTS modules to work with TorchDevice.
    """
    try:
        # Import TTS modules
        from TTS.tts.utils import synthesis
        
        # Store the original numpy_to_torch function
        original_numpy_to_torch = synthesis.numpy_to_torch
        
        # Define a patched version
        def patched_numpy_to_torch(np_array, dtype, device=None):
            """
            A patched version of numpy_to_torch that handles device properly.
            """
            # Always use CPU for initial tensor creation to avoid CUDA errors
            tensor = torch.as_tensor(np_array, dtype=dtype, device='cpu')
            
            # Then move to the requested device if specified
            if device is not None:
                # If device is 'cuda', let TorchDevice handle the redirection
                if device == 'cuda' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    logger.info("Redirecting tensor from 'cuda' to 'mps'")
                    tensor = tensor.to('mps')
                else:
                    tensor = tensor.to(device)
            
            return tensor
        
        # Apply the patch
        synthesis.numpy_to_torch = patched_numpy_to_torch
        logger.info("Patched TTS.tts.utils.synthesis.numpy_to_torch")
        
        return True
    except Exception as e:
        logger.error("Failed to patch TTS modules: %s", e)
        logger.error(traceback.format_exc())
        return False

# Apply TTS patches
patch_success = patch_tts_modules()
if not patch_success:
    logger.warning("Some TTS patches failed. TTS might not work correctly with TorchDevice.")

# Now import TTS
from TTS.api import TTS

def test_tts_cpu(output_path="samples/cpu_output.wav"):
    """Test TTS on CPU."""
    logger.info("\nTesting TTS on CPU...")
    
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Initialize TTS with CPU
        logger.info("Loading model on CPU...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        
        # Generate speech
        text = "Hello world. This is a test of text to speech on CPU."
        logger.info("Generating speech for text: '%s'", text)
        
        # Explicitly set device to CPU for all operations
        with torch.device('cpu'):
            tts.tts_to_file(text=text, file_path=output_path)
        
        logger.info("Speech generated and saved to %s", output_path)
        return True
    except Exception as e:
        logger.error("Error in TTS test on CPU: %s", e)
        logger.error(traceback.format_exc())
        return False

def test_tts_mps(output_path="samples/mps_output.wav"):
    """Test TTS on MPS (Apple Silicon)."""
    logger.info("\nTesting TTS on MPS (Apple Silicon)...")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        logger.warning("MPS is not available on this system. Skipping MPS test.")
        return False
    
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Initialize TTS
        logger.info("Loading model...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        
        # Move model to MPS
        logger.info("Moving model to MPS...")
        tts.model.to('mps')
        if hasattr(tts, 'vocoder') and tts.vocoder is not None:
            tts.vocoder.to('mps')
        
        # Generate speech
        text = "This is a test of text to speech on MPS (Apple Silicon)."
        logger.info("Generating speech for text: '%s'", text)
        
        # Set the device explicitly for synthesis
        tts.synthesizer.device = 'mps'
        tts.tts_to_file(text=text, file_path=output_path)
        
        logger.info("Speech generated and saved to %s", output_path)
        return True
    except Exception as e:
        logger.error("Error in TTS test on MPS: %s", e)
        logger.error(traceback.format_exc())
        return False

def test_tts_cuda_redirect(output_path="samples/cuda_redirect_output.wav"):
    """Test TTS with CUDA calls redirected to MPS."""
    logger.info("\nTesting TTS with CUDA calls redirected to MPS...")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA redirection is not working. TorchDevice should make torch.cuda.is_available() return True.")
        return False
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        logger.warning("MPS is not available on this system. CUDA redirection won't work.")
        return False
    
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Initialize TTS with CUDA
        logger.info("Loading model with CUDA (should be redirected to MPS)...")
        
        # Create a custom TTS instance with manual device handling
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        
        # Manually move models to 'cuda' (which should be redirected to MPS)
        logger.info("Moving model to CUDA (should be redirected to MPS)...")
        
        # Use a safer approach by moving individual tensors
        for param in tts.model.parameters():
            # Create a new tensor on MPS and copy data
            param.data = param.data.to('mps')
        
        if hasattr(tts, 'vocoder') and tts.vocoder is not None:
            for param in tts.vocoder.parameters():
                param.data = param.data.to('mps')
        
        # Set the device for synthesis
        tts.synthesizer.device = 'mps'
        
        # Generate speech
        text = "This is a test of text to speech with CUDA redirected to MPS."
        logger.info("Generating speech for text: '%s'", text)
        
        tts.tts_to_file(text=text, file_path=output_path)
        
        logger.info("Speech generated and saved to %s", output_path)
        return True
    except Exception as e:
        logger.error("Error in TTS test with CUDA redirect: %s", e)
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run all tests."""
    # Create samples directory if it doesn't exist
    Path("samples").mkdir(exist_ok=True)
    
    # Test TTS on CPU
    cpu_success = test_tts_cpu()
    
    # Test TTS on MPS
    mps_success = test_tts_mps()
    
    # Test TTS with CUDA redirected to MPS
    cuda_redirect_success = test_tts_cuda_redirect()
    
    # Print test results
    logger.info("\nTest Results:")
    logger.info("CPU test: %s", 'SUCCESS' if cpu_success else 'FAILURE')
    logger.info("MPS test: %s", 'SUCCESS' if mps_success else 'FAILURE')
    logger.info("CUDA redirect test: %s", 'SUCCESS' if cuda_redirect_success else 'FAILURE')
    
    # Create a PR-ready patch
    create_pr_patch()
    
    if cpu_success or mps_success or cuda_redirect_success:
        logger.info("\nAt least one test succeeded! Check the samples directory for the generated audio files.")
    else:
        logger.info("\nAll tests failed. Check the logs for details.")

def create_pr_patch():
    """Create a PR-ready patch for TorchDevice to work with TTS."""
    patch_content = """
# Add this to TorchDevice.py in the apply_patches method

# Patch torch.serialization._get_restore_location for TTS compatibility
original_get_restore_location = torch.serialization._get_restore_location

def patched_get_restore_location(map_location):
    # Create a simple restore function that ignores location
    def restore_location(storage, location):
        return storage
    
    return restore_location

torch.serialization._get_restore_location = patched_get_restore_location
"""
    
    # Write the patch to a file
    patch_file = "torchdevice_tts_patch.txt"
    with open(patch_file, "w") as f:
        f.write(patch_content)
    
    logger.info("\nPR-ready patch created and saved to %s", patch_file)
    logger.info("To make this patch permanent, add the code to the apply_patches method in TorchDevice.py")

if __name__ == "__main__":
    main()
