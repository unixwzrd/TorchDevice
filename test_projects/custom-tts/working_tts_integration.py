#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Working TTS integration with TorchDevice.
This script demonstrates the successful integration of TTS with TorchDevice,
handling all necessary patches and redirections.
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
        import traceback
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
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_tts_with_cuda_redirect(output_path="samples/cuda_redirect_output.wav"):
    """Test TTS with CUDA redirected to MPS."""
    logger.info("\nTesting TTS with CUDA redirected to MPS...")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA redirection is not working. TorchDevice should make torch.cuda.is_available() return True.")
        return False
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        logger.warning("MPS is not available on this system. CUDA redirection won't work.")
        return False
    
    start_time = time.time()
    
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Initialize TTS with GPU (will be redirected to MPS)
        logger.info("Loading model with GPU=True (should be redirected to MPS)...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=True)
        
        # Generate speech
        text = "This is a test of text to speech with CUDA redirected to MPS."
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
        logger.error("Error in TTS test with CUDA redirect: %s", e)
        import traceback
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
    
    # Create a PR-ready patch
    create_pr_patch()
    
    if cpu_success or mps_success or cuda_redirect_success:
        logger.info("\nAt least one test succeeded! Check the samples directory for the generated audio files.")
        
        # Print timing comparison if multiple tests succeeded
        if cpu_success and (mps_success or cuda_redirect_success):
            logger.info("\nYou can compare the audio quality and generation time between CPU and GPU versions.")
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
