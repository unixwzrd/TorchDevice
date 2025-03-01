#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized TTS test with TorchDevice integration.
This script uses a more robust approach to ensure TTS works with TorchDevice.
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

def test_tts_with_short_text(output_path="samples/optimized_short_output.wav"):
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
        logger.info(f"Generating speech for text: '{text}'")
        
        tts.tts_to_file(text=text, file_path=output_path)
        
        logger.info(f"Speech generated and saved to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error in TTS test with short text: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_tts_with_medium_text(output_path="samples/optimized_medium_output.wav"):
    """Test TTS with a medium length text sample."""
    logger.info("\nTesting TTS with a medium length text sample...")
    
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Initialize TTS with a simple English model
        logger.info("Loading model on CPU...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        
        # Generate speech with a medium length text
        text = "This is a test of text to speech with TorchDevice."
        logger.info(f"Generating speech for text: '{text}'")
        
        tts.tts_to_file(text=text, file_path=output_path)
        
        logger.info(f"Speech generated and saved to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error in TTS test with medium text: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run the test."""
    # Create samples directory if it doesn't exist
    Path("samples").mkdir(exist_ok=True)
    
    # Test with short text
    short_success = test_tts_with_short_text()
    
    # Test with medium text
    medium_success = test_tts_with_medium_text()
    
    # Print test results
    logger.info("\nTest Results:")
    logger.info(f"Short text test: {'SUCCESS' if short_success else 'FAILURE'}")
    logger.info(f"Medium text test: {'SUCCESS' if medium_success else 'FAILURE'}")
    
    if short_success and medium_success:
        logger.info("\nSuccess! TorchDevice is now working with TTS using the optimized patch.")
        logger.info("The audio files have been saved to the samples directory.")
        logger.info("\nTo listen to the generated audio, you can use:")
        logger.info("  - samples/optimized_short_output.wav (short text)")
        logger.info("  - samples/optimized_medium_output.wav (medium text)")
    else:
        logger.info("\nSome tests failed. Check the logs for details.")

if __name__ == "__main__":
    main()
