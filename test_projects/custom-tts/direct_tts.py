#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct TTS with TorchDevice.
This script demonstrates using Coqui TTS directly without patching.
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Now import TTS
from TTS.api import TTS

def print_device_info():
    """Print information about the available devices."""
    logger.info("============================================================")
    logger.info("Direct TTS without TorchDevice")
    logger.info("============================================================")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name()}")
    
    # Check for MPS
    if hasattr(torch, 'mps') and torch.mps.is_available():
        logger.info("MPS (Apple Silicon) is available")
    else:
        logger.info("MPS (Apple Silicon) is not available")
    
    logger.info("============================================================")

def test_basic_tts(output_path="samples/direct_output.wav"):
    """Test basic TTS functionality with a simple model."""
    logger.info("\nTesting basic TTS with a simple model...")
    
    start_time = time.time()
    
    try:
        # Initialize TTS with a simple English model
        # Force CPU for model loading
        logger.info("Loading model on CPU...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        
        # Generate speech
        text = "Hello, this is a test of direct text to speech without TorchDevice."
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

def test_multilingual_tts(output_path="samples/direct_multilingual_output.wav"):
    """Test multilingual TTS functionality."""
    logger.info("\nTesting multilingual TTS...")
    
    start_time = time.time()
    
    try:
        # Initialize TTS with a multilingual model
        logger.info("Loading multilingual model on CPU...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        
        # Generate speech in Spanish
        text = "Hola, esta es una prueba de síntesis de voz multilingüe sin TorchDevice."
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
    
    # Print device information
    print_device_info()
    
    # Test basic TTS
    basic_result = test_basic_tts()
    
    # Test multilingual TTS
    multilingual_result = test_multilingual_tts()
    
    # Print test results
    logger.info("\nTest Results:")
    logger.info(f"Basic TTS: {'SUCCESS' if basic_result else 'FAILURE'}")
    logger.info(f"Multilingual TTS: {'SUCCESS' if multilingual_result else 'FAILURE'}")

if __name__ == "__main__":
    main()
