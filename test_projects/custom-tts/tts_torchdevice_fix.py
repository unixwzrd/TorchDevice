#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS with TorchDevice Fix.
This script applies a targeted fix to make TTS work with TorchDevice.
"""

import os
import sys
import time
import torch
import logging
import functools
from pathlib import Path

# Add the TorchDevice module to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import TorchDevice first
import TorchDevice

# Store original functions for later restoration
_original_get_restore_location = torch.serialization._get_restore_location

# Create a patched version of _get_restore_location
@functools.wraps(_original_get_restore_location)
def patched_get_restore_location(map_location):
    """
    A patched version of torch.serialization._get_restore_location that handles
    TorchDevice device objects correctly.
    """
    logger.info("Patched _get_restore_location called with map_location: %s", map_location)
    
    # Handle device objects by checking for attributes instead of using isinstance
    if hasattr(map_location, 'type') and hasattr(map_location, 'index'):
        logger.info("Detected device object with type: %s", map_location.type)
        
        # Create a restore function that handles device mapping
        def restore_location(storage, location):
            logger.info("Restore location called with location: %s", location)
            # Just return the storage as is - we'll let TorchDevice handle device mapping
            return storage
        
        return restore_location
    
    # Fall back to original function for other cases
    return _original_get_restore_location(map_location)

def apply_patches():
    """Apply patches to make TTS work with TorchDevice."""
    logger.info("Applying TTS compatibility patches")
    
    # Patch torch.serialization._get_restore_location
    torch.serialization._get_restore_location = patched_get_restore_location
    
    logger.info("TTS compatibility patches applied")

def remove_patches():
    """Remove patches applied by apply_patches()."""
    logger.info("Removing TTS compatibility patches")
    
    # Restore torch.serialization._get_restore_location
    torch.serialization._get_restore_location = _original_get_restore_location
    
    logger.info("TTS compatibility patches removed")

# Apply patches immediately
apply_patches()

# Now import TTS
from TTS.api import TTS

def test_basic_tts(output_path="samples/fixed_output.wav"):
    """Test basic TTS functionality with a simple model."""
    logger.info("\nTesting basic TTS with a simple model...")
    
    start_time = time.time()
    
    try:
        # Initialize TTS with a simple English model
        # Force CPU for model loading to avoid device issues
        logger.info("Loading model on CPU...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        
        # Generate speech
        text = "Hello, this is a test of text to speech with the fixed TorchDevice integration."
        logger.info("Generating speech for text: '%s'", text)
        
        tts.tts_to_file(text=text, file_path=output_path)
        
        end_time = time.time()
        logger.info("Speech generated and saved to %s", output_path)
        logger.info("Time taken: %.2f seconds", end_time - start_time)
        
        return os.path.exists(output_path)
    except Exception as e:
        logger.error("Error in basic TTS test: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_multilingual_tts(output_path="samples/fixed_multilingual_output.wav"):
    """Test multilingual TTS functionality."""
    logger.info("\nTesting multilingual TTS...")
    
    start_time = time.time()
    
    try:
        # Initialize TTS with a multilingual model
        logger.info("Loading multilingual model on CPU...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        
        # Generate speech in Spanish
        text = "Hola, esta es una prueba de síntesis de voz multilingüe con TorchDevice."
        language = "es"
        logger.info("Generating speech for text in %s: '%s'", language, text)
        
        tts.tts_to_file(text=text, file_path=output_path, language=language)
        
        end_time = time.time()
        logger.info("Multilingual speech generated and saved to %s", output_path)
        logger.info("Time taken: %.2f seconds", end_time - start_time)
        
        return os.path.exists(output_path)
    except Exception as e:
        logger.error("Error in multilingual TTS test: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run the test."""
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Test basic TTS
        basic_result = test_basic_tts()
        
        # Test multilingual TTS if basic test succeeds
        multilingual_result = False
        if basic_result:
            multilingual_result = test_multilingual_tts()
        
        # Print test results
        logger.info("\nTest Results:")
        logger.info("Basic TTS: %s", 'SUCCESS' if basic_result else 'FAILURE')
        logger.info("Multilingual TTS: %s", 'SUCCESS' if multilingual_result else 'FAILURE')
    finally:
        # Always remove patches at the end
        remove_patches()

if __name__ == "__main__":
    main()
