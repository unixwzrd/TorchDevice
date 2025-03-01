#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patched Integrated TTS with TorchDevice.
This script demonstrates using TTS with TorchDevice integration
using the TorchDevice TTS patch.
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

# Add the TorchDevice module to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import TorchDevice first to ensure it's loaded before any torch imports
import TorchDevice

# Import our TTS patch
import torchdevice_tts_patch

# Now import torch and TTS
import torch
from TTS.api import TTS

def test_basic_tts(output_path="samples/patched_output.wav"):
    """Test basic TTS functionality with a simple model."""
    logger.info("\nTesting basic TTS with a simple model...")
    
    start_time = time.time()
    
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Initialize TTS with a simple English model
        # We'll use CPU for model loading to avoid device issues
        logger.info("Loading model on CPU...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        
        # Generate speech
        text = "Hello, this is a test of text to speech with the patched TorchDevice integration."
        logger.info("Generating speech for text: '%s'", text)
        
        tts.tts_to_file(text=text, file_path=output_path)
        
        end_time = time.time()
        logger.info("Speech generated and saved to %s", output_path)
        logger.info("Time taken: %.2f seconds", end_time - start_time)
        
        return True
    except Exception as e:
        logger.error("Error in basic TTS test: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run the test."""
    try:
        # Test basic TTS
        success = test_basic_tts()
        
        # Print test results
        logger.info("\nTest Results:")
        logger.info("Basic TTS: %s", 'SUCCESS' if success else 'FAILURE')
        
        if success:
            logger.info("\nSuccess! TorchDevice is now working with TTS.")
            logger.info("The audio file has been saved to samples/patched_output.wav")
    finally:
        # Always remove patches at the end
        torchdevice_tts_patch.remove_tts_patches()

if __name__ == "__main__":
    main()
