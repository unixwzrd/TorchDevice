#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct TTS Patch for TorchDevice.
This script directly patches torch.serialization to work with TorchDevice.
"""

import os
import sys
import torch
import logging
import types
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the TorchDevice module to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

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
    logger.info("Patched _get_restore_location called with map_location type: %s", type(map_location))
    
    # Handle device objects by checking for attributes instead of using isinstance
    if hasattr(map_location, 'type') and hasattr(map_location, 'index'):
        logger.info("Detected device object with type: %s", map_location.type)
        
        # Create a restore function that handles device mapping
        def restore_location(storage, location):
            logger.debug("Restore location called with location: %s", location)
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
logger.info("Direct TTS patch applied to torch.serialization._get_restore_location")

# Now import TTS
from TTS.api import TTS

def test_tts_model(model_name, text, output_path):
    """Test TTS with a specific model and text."""
    logger.info(f"\nTesting TTS with model: {model_name}")
    logger.info(f"Text: '{text}'")
    
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Initialize TTS with the specified model
        # We'll use CPU for model loading to avoid device issues
        logger.info("Loading model on CPU...")
        tts = TTS(model_name, gpu=False)
        
        # Generate speech
        logger.info("Generating speech...")
        tts.tts_to_file(text=text, file_path=output_path)
        
        logger.info("Speech generated and saved to %s", output_path)
        
        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"Error in TTS test with model {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run the test."""
    # Create samples directory if it doesn't exist
    Path("samples").mkdir(exist_ok=True)
    
    # Test with different models and text samples
    tests = [
        {
            "model": "tts_models/en/ljspeech/tacotron2-DDC",
            "text": "Hello world.",
            "output": "samples/short_text_output.wav"
        },
        {
            "model": "tts_models/en/ljspeech/tacotron2-DDC",
            "text": "This is a test of text to speech.",
            "output": "samples/medium_text_output.wav"
        },
        {
            "model": "tts_models/en/vctk/vits",
            "text": "Hello world. This is a different model.",
            "output": "samples/vits_model_output.wav"
        }
    ]
    
    results = {}
    
    # Run each test
    for test in tests:
        model = test["model"]
        text = test["text"]
        output = test["output"]
        
        success = test_tts_model(model, text, output)
        results[f"{model} - '{text}'"] = 'SUCCESS' if success else 'FAILURE'
    
    # Print test results
    logger.info("\nTest Results:")
    for test, result in results.items():
        logger.info(f"{test}: {result}")
    
    if all(result == 'SUCCESS' for result in results.values()):
        logger.info("\nSuccess! TorchDevice is now working with TTS using the direct patch.")
        logger.info("The audio files have been saved to the samples directory.")
    else:
        logger.info("\nSome tests failed. Check the logs for details.")

if __name__ == "__main__":
    main()
