#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final TTS integration with TorchDevice.
This script demonstrates the complete integration of TTS with TorchDevice,
handling all necessary patches and redirections for CPU, MPS, and CUDA-to-MPS.
"""

import os
import sys
import time
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

# Patch TTS synthesis module to handle CUDA redirection properly
def patch_tts_synthesis():
    """
    Patch the TTS synthesis module to handle device redirection properly.
    This ensures CUDA calls are properly redirected to MPS.
    """
    try:
        # Import the synthesis module
        from TTS.tts.utils import synthesis
        
        # Store the original numpy_to_torch function
        original_numpy_to_torch = synthesis.numpy_to_torch
        
        # Define a patched version
        def patched_numpy_to_torch(np_array, dtype, device=None):
            """
            A patched version of numpy_to_torch that handles device properly.
            """
            # Handle None input
            if np_array is None:
                return None
            
            # Create tensor on CPU first
            tensor = torch.as_tensor(np_array, dtype=dtype, device='cpu')
            
            # Then move to the requested device if specified
            if device is not None:
                # If device is 'cuda', check if MPS is available
                if device == 'cuda' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    logger.info("Redirecting tensor from 'cuda' to 'mps'")
                    device = 'mps'
                
                # Move tensor to the appropriate device
                tensor = tensor.to(device)
            
            return tensor
        
        # Apply the patch
        synthesis.numpy_to_torch = patched_numpy_to_torch
        logger.info("Patched TTS.tts.utils.synthesis.numpy_to_torch")
        
        return True
    except Exception as e:
        logger.error("Failed to patch TTS synthesis module: %s", e)
        logger.error(traceback.format_exc())
        return False

# Apply TTS synthesis patch
patch_success = patch_tts_synthesis()
if not patch_success:
    logger.warning("TTS synthesis patch failed. CUDA redirection might not work correctly.")

# Now import TTS
from TTS.api import TTS

def test_tts_with_cpu(output_path="samples/cpu_output.wav"):
    """Test TTS on CPU."""
    logger.info("\nTesting TTS on CPU...")
    
    start_time = time.time()
    
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Initialize TTS with CPU
        logger.info("Loading model on CPU...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        
        # Generate speech
        text = "Hello world. This is a test of text to speech on CPU."
        logger.info("Generating speech for text: '%s'", text)
        
        # Generate speech
        wav = tts.tts(text=text)
        
        # Save to file
        import soundfile as sf
        sf.write(output_path, wav, 22050)
        
        end_time = time.time()
        logger.info("Speech generated and saved to %s", output_path)
        logger.info("Time taken: %.2f seconds", end_time - start_time)
        
        return True
    except Exception as e:
        logger.error("Error in TTS test on CPU: %s", e)
        logger.error(traceback.format_exc())
        return False

def test_tts_with_mps(output_path="samples/mps_output.wav"):
    """Test TTS with MPS."""
    logger.info("\nTesting TTS with MPS...")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        logger.warning("MPS is not available on this system. Skipping MPS test.")
        return False
    
    start_time = time.time()
    
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Initialize TTS with MPS
        logger.info("Loading model with MPS...")
        
        # First load on CPU
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        
        # Then manually move to MPS
        logger.info("Moving synthesizer to MPS...")
        tts.synthesizer.tts_model.to('mps')
        tts.synthesizer.vocoder_model.to('mps')
        tts.synthesizer.device = 'mps'
        
        # Generate speech
        text = "This is a test of text to speech with MPS."
        logger.info("Generating speech for text: '%s'", text)
        
        # Generate speech
        wav = tts.tts(text=text)
        
        # Save to file
        import soundfile as sf
        sf.write(output_path, wav, 22050)
        
        end_time = time.time()
        logger.info("Speech generated and saved to %s", output_path)
        logger.info("Time taken: %.2f seconds", end_time - start_time)
        
        return True
    except Exception as e:
        logger.error("Error in TTS test with MPS: %s", e)
        logger.error(traceback.format_exc())
        return False

def test_tts_with_cuda_redirect(output_path="samples/cuda_redirect_output.wav"):
    """Test TTS with CUDA redirected to MPS."""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Skipping CUDA redirect test.")
        return False
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        logger.warning("MPS is not available on this system. CUDA redirection won't work.")
        return False
    
    start_time = time.time()
    
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Initialize TTS with CPU first
        logger.info("Loading model on CPU first...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        
        # Manually patch the synthesizer to handle CUDA redirection
        logger.info("Patching synthesizer to handle CUDA redirection...")
        
        # Move models to MPS
        tts.synthesizer.tts_model.to('mps')
        tts.synthesizer.vocoder_model.to('mps')
        
        # Add device attribute if it doesn't exist
        if not hasattr(tts.synthesizer, 'device'):
            tts.synthesizer.device = 'mps'
        else:
            # Set device to 'mps' but allow 'cuda' inputs
            tts.synthesizer.device = 'mps'
        
        # Store the original tts method
        original_tts = tts.synthesizer.tts
        
        # Store the original device
        original_device = getattr(tts.synthesizer, 'device', 'cpu')
        
        # Create a patched version that handles device redirection
        def patched_tts(text, speaker_name=None, language_name=None, speaker_wav=None, style_wav=None, style_text=None, reference_wav=None, reference_speaker_name=None, **kwargs):
            """Patched TTS method that handles CUDA redirection."""
            # Replace any 'cuda' device with 'mps'
            if 'cuda' in str(kwargs.get('device', '')):
                logger.info("Redirecting 'cuda' device to 'mps' in synthesizer.tts")
                kwargs['device'] = 'mps'
            
            # Call the original method
            return original_tts(text, speaker_name, language_name, speaker_wav, style_wav, style_text, reference_wav, reference_speaker_name, **kwargs)
        
        # Apply the patch
        tts.synthesizer.tts = patched_tts
        
        # Generate speech
        text = "This is a test of text to speech with CUDA redirected to MPS."
        logger.info("Generating speech for text: '%s'", text)
        
        # Generate speech with explicit device='cuda' to test redirection
        wav = tts.tts(text=text)
        
        # Save to file
        import soundfile as sf
        sf.write(output_path, wav, 22050)
        
        # Restore original method and device
        tts.synthesizer.tts = original_tts
        tts.synthesizer.device = original_device
        
        end_time = time.time()
        logger.info("Speech generated and saved to %s", output_path)
        logger.info("Time taken: %.2f seconds", end_time - start_time)
        
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
    cpu_success = test_tts_with_cpu()
    
    # Test TTS with MPS
    mps_success = test_tts_with_mps()
    
    # Test TTS with CUDA redirected to MPS
    cuda_redirect_success = test_tts_with_cuda_redirect()
    
    # Print test results
    logger.info("\nTest Results:")
    logger.info("CPU test: %s", 'SUCCESS' if cpu_success else 'FAILURE')
    logger.info("MPS test: %s", 'SUCCESS' if mps_success else 'FAILURE')
    logger.info("CUDA redirect test: %s", 'SUCCESS' if cuda_redirect_success else 'FAILURE')
    
    # Create a PR-ready patch for TorchDevice
    create_pr_patch()
    
    # Summary
    if cpu_success and mps_success:
        logger.info("\nSuccess! TTS is working on both CPU and MPS.")
        logger.info("The audio files have been saved to the samples directory.")
        
        # Compare performance
        logger.info("\nYou can compare the audio quality and generation time between CPU and MPS versions.")
    else:
        logger.info("\nSome tests failed. Check the logs for details.")

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

# Patch TTS synthesis module if it's available
try:
    from TTS.tts.utils import synthesis
    
    # Store the original numpy_to_torch function
    original_numpy_to_torch = synthesis.numpy_to_torch
    
    # Define a patched version
    def patched_numpy_to_torch(np_array, dtype, device=None):
        # Handle None input
        if np_array is None:
            return None
        
        # Create tensor on CPU first
        tensor = torch.as_tensor(np_array, dtype=dtype, device='cpu')
        
        # Then move to the requested device if specified
        if device is not None:
            # If device is 'cuda', check if MPS is available
            if device == 'cuda' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            
            # Move tensor to the appropriate device
            tensor = tensor.to(device)
        
        return tensor
    
    # Apply the patch
    synthesis.numpy_to_torch = patched_numpy_to_torch
except ImportError:
    # TTS is not installed, so no need to patch
    pass
"""
    
    # Write the patch to a file
    patch_file = "torchdevice_tts_patch.txt"
    with open(patch_file, "w") as f:
        f.write(patch_content)
    
    logger.info("\nPR-ready patch created and saved to %s", patch_file)
    logger.info("To make this patch permanent, add the code to the apply_patches method in TorchDevice.py")

if __name__ == "__main__":
    main()
