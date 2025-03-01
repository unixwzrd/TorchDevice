#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS with TorchDevice.
This script demonstrates using Coqui TTS with TorchDevice handling device detection.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add the TorchDevice module to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# First import TorchDevice to handle device detection
import TorchDevice

# Now import torch and TTS
import torch
from TTS.api import TTS

def test_basic_tts(output_path="samples/torchdevice_output.wav"):
    """Test basic TTS functionality with a simple model."""
    logger.info("\nTesting basic TTS with a simple model...")
    
    start_time = time.time()
    
    try:
        # Initialize TTS with a simple English model
        # Let TorchDevice handle the device selection
        logger.info("Loading model with TorchDevice handling device selection...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        
        # Generate speech
        text = "Hello, this is a test of text to speech with TorchDevice."
        logger.info(f"Generating speech for text: '{text}'")
        
        tts.tts_to_file(text=text, file_path=output_path)
        
        end_time = time.time()
        logger.info(f"Speech generated and saved to {output_path}")
        logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
        
        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"Error in basic TTS test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_multilingual_tts(output_path="samples/torchdevice_multilingual_output.wav"):
    """Test multilingual TTS functionality."""
    logger.info("\nTesting multilingual TTS...")
    
    start_time = time.time()
    
    try:
        # Initialize TTS with a multilingual model
        logger.info("Loading multilingual model with TorchDevice handling device selection...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Generate speech in Spanish
        text = "Hola, esta es una prueba de síntesis de voz multilingüe con TorchDevice."
        language = "es"
        logger.info(f"Generating speech for text in {language}: '{text}'")
        
        tts.tts_to_file(text=text, file_path=output_path, language=language)
        
        end_time = time.time()
        logger.info(f"Multilingual speech generated and saved to {output_path}")
        logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
        
        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"Error in multilingual TTS test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run the test."""
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
    logger.info(f"Basic TTS: {'SUCCESS' if basic_result else 'FAILURE'}")
    logger.info(f"Multilingual TTS: {'SUCCESS' if multilingual_result else 'FAILURE'}")

if __name__ == "__main__":
    main()
