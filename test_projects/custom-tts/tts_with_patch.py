#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS with TorchDevice using compatibility patches.
This script demonstrates using Coqui TTS with TorchDevice by applying compatibility patches.
"""

import os
import sys
import time
import torch
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

# First import TorchDevice
import TorchDevice

# Import our patch and apply it
from tts_patch import apply_patches, remove_patches
apply_patches()

# Now import TTS
from TTS.api import TTS

def print_device_info():
    """Print information about the available devices."""
    logger.info("============================================================")
    logger.info("TTS with TorchDevice and Patches")
    logger.info("============================================================")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Device count: {torch.cuda.device_count()}")
    logger.info(f"Current device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name()}")
    
    # Get the default device
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Default device: {default_device}")
    logger.info("============================================================")

def test_basic_tts(output_path="samples/patched_output.wav"):
    """Test basic TTS functionality with a simple model."""
    logger.info("\nTesting basic TTS with a simple model...")
    
    start_time = time.time()
    
    try:
        # Initialize TTS with a simple English model
        # Force CPU for model loading
        logger.info("Loading model on CPU...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        
        # Generate speech
        text = "Hello, this is a test of the patched TorchDevice library with text to speech."
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
    try:
        # Create samples directory if it doesn't exist
        Path("samples").mkdir(exist_ok=True)
        
        # Print device information
        print_device_info()
        
        # Test basic TTS
        basic_result = test_basic_tts()
        
        # Print test results
        logger.info("\nTest Results:")
        logger.info(f"Basic TTS: {'SUCCESS' if basic_result else 'FAILURE'}")
    finally:
        # Always remove patches at the end
        remove_patches()

if __name__ == "__main__":
    main()
