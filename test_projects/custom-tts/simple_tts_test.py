#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple TTS test without TorchDevice integration.
This script demonstrates using Coqui TTS without TorchDevice.
"""

import os
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import TTS
from TTS.api import TTS

def test_basic_tts(output_path="samples/simple_output.wav"):
    """Test basic TTS functionality with a simple model."""
    logger.info("\nTesting basic TTS with a simple model...")
    
    start_time = time.time()
    
    try:
        # Initialize TTS with a simple English model
        logger.info("Loading model on CPU...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        
        # Generate speech
        text = "Hello, this is a test of text to speech without TorchDevice."
        logger.info(f"Generating speech for text: '{text}'")
        
        tts.tts_to_file(text=text, file_path=output_path)
        
        end_time = time.time()
        logger.info(f"Speech generated and saved to {output_path}")
        logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
        
        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"Error in basic TTS test: {e}")
        return False

def main():
    """Main function to run the test."""
    # Create samples directory if it doesn't exist
    Path("samples").mkdir(exist_ok=True)
    
    # Test basic TTS
    basic_result = test_basic_tts()
    
    # Print test results
    logger.info("\nTest Results:")
    logger.info(f"Basic TTS: {'SUCCESS' if basic_result else 'FAILURE'}")

if __name__ == "__main__":
    main()
