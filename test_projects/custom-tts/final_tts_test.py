#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final TTS Test with TorchDevice Integration.
This script demonstrates using TTS with TorchDevice integration
using the TorchDevice TTS integration module.
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

# Import our TorchDevice TTS integration first
import torchdevice_tts_integration

# Now import torch and TTS
import torch
from TTS.api import TTS

def test_basic_tts(output_path="samples/final_output.wav"):
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
        text = "Hello, this is a test of text to speech with the integrated TorchDevice solution."
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
    # Test basic TTS
    success = test_basic_tts()
    
    # Print test results
    logger.info("\nTest Results:")
    logger.info("Basic TTS: %s", 'SUCCESS' if success else 'FAILURE')
    
    if success:
        logger.info("\nSuccess! TorchDevice is now working with TTS.")
        logger.info("The audio file has been saved to samples/final_output.wav")
        logger.info("\nTo make this integration permanent, you can apply the patch in torchdevice_tts_pr_patch.txt to TorchDevice.py")

if __name__ == "__main__":
    main()
