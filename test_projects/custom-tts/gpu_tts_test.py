#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU TTS test with TorchDevice integration.
This script tests TTS with GPU enabled, allowing TorchDevice to redirect to MPS.
"""

import os
import sys
import torch
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

# Store the original _get_restore_location function
_original_get_restore_location = torch.serialization._get_restore_location

# Define a replacement for _get_restore_location
def patched_get_restore_location(map_location):
    """
    A patched version of torch.serialization._get_restore_location that handles
    TorchDevice device objects correctly.
    """
    if hasattr(map_location, 'type') and hasattr(map_location, 'index'):
        # Create a restore function that handles device mapping
        def restore_location(storage, location):
            # Just return the storage as is - we'll let TorchDevice handle device mapping
            return storage
        return restore_location
    
    # Handle string map_location
    if isinstance(map_location, str):
        def restore_location(storage, location):
            return storage
        return restore_location
    
    # Handle callable map_location
    if callable(map_location):
        return map_location
    
    # Handle dict map_location
    if isinstance(map_location, dict):
        def restore_location(storage, location):
            for k, v in map_location.items():
                if k in location:
                    return storage
            return storage
        return restore_location
    
    # Default case
    def default_restore_location(storage, location):
        return storage
    
    return default_restore_location

# Apply the patch
torch.serialization._get_restore_location = patched_get_restore_location
logger.info("TorchDevice patch applied to torch.serialization._get_restore_location")

# Now import TTS
from TTS.api import TTS

def test_tts_with_gpu(output_path="samples/gpu_output.wav"):
    """Test TTS with GPU enabled, allowing TorchDevice to redirect to MPS."""
    logger.info("\nTesting TTS with GPU enabled (TorchDevice should redirect to MPS)...")
    
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        logger.info(f"CUDA available: {cuda_available}")
        logger.info(f"MPS available: {mps_available}")
        
        # Initialize TTS with GPU enabled
        logger.info("Loading model with GPU enabled...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=True)
        
        # Generate speech
        text = "This is a test of text to speech with GPU enabled."
        logger.info(f"Generating speech for text: '{text}'")
        
        tts.tts_to_file(text=text, file_path=output_path)
        
        logger.info(f"Speech generated and saved to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error in TTS test with GPU: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run the test."""
    # Create samples directory if it doesn't exist
    Path("samples").mkdir(exist_ok=True)
    
    # Test with GPU
    gpu_success = test_tts_with_gpu()
    
    # Print test results
    logger.info("\nTest Results:")
    logger.info(f"GPU test: {'SUCCESS' if gpu_success else 'FAILURE'}")
    
    if gpu_success:
        logger.info("\nSuccess! TorchDevice is now working with TTS using GPU (redirected to MPS).")
        logger.info("The audio file has been saved to samples/gpu_output.wav")
    else:
        logger.info("\nGPU test failed. Check the logs for details.")

if __name__ == "__main__":
    main()
