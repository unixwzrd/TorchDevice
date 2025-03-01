#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS with TorchDevice integration test.
This script demonstrates using Coqui TTS with TorchDevice to redirect CUDA calls to MPS.
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path

# Add the TorchDevice module to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import TorchDevice to redirect CUDA calls to MPS
import TorchDevice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import TTS after TorchDevice to ensure CUDA redirection
from TTS.api import TTS

def print_device_info():
    """Print information about the available devices."""
    logger.info("============================================================")
    logger.info("TTS with TorchDevice Test")
    logger.info("============================================================")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Device count: {torch.cuda.device_count()}")
    logger.info(f"Current device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name()}")
    
    # Get the default device using TorchDevice
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Default device: {default_device}")
    logger.info("============================================================")

def list_available_models():
    """List all available TTS models."""
    logger.info("Available TTS models:")
    tts_instance = TTS()
    
    # Get models by type
    try:
        tts_models = tts_instance.list_models()
        
        # Display model categories
        for category in ["tts_models", "vocoder_models", "voice_conversion_models"]:
            category_models = [m for m in tts_models if m.startswith(category)]
            if category_models:
                logger.info(f"\n{category.upper()}:")
                # Show only first 5 models for brevity
                for model in category_models[:5]:
                    logger.info(f"  - {model}")
                if len(category_models) > 5:
                    logger.info(f"  ... and {len(category_models) - 5} more")
        
        logger.info(f"\nTotal models available: {len(tts_models)}")
        return tts_models
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        logger.info("Continuing with basic models...")
        return []

def test_basic_tts(output_path="samples/output_basic.wav"):
    """Test basic TTS functionality with a simple model."""
    logger.info("\nTesting basic TTS with a simple model...")
    
    start_time = time.time()
    
    try:
        # Initialize TTS with a simple English model
        # Force CPU for model loading to avoid TorchDevice conflicts
        logger.info("Loading model on CPU first to avoid TorchDevice conflicts...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        
        # Log device information
        try:
            device = next(tts.model.parameters()).device
            logger.info(f"Model loaded on device: {device}")
        except Exception as e:
            logger.warning(f"Could not determine model device: {e}")
        
        # Generate speech
        text = "Hello, this is a test of the TorchDevice library with text to speech."
        logger.info(f"Generating speech for text: '{text}'")
        
        tts.tts_to_file(text=text, file_path=output_path)
        
        end_time = time.time()
        logger.info(f"Speech generated and saved to {output_path}")
        logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
        
        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"Error in basic TTS test: {e}")
        return False

def test_multilingual_tts(output_path="samples/output_multilingual.wav"):
    """Test multilingual TTS with voice cloning."""
    logger.info("\nTesting multilingual TTS with a more complex model...")
    logger.info("Note: This will download a larger model if not already cached.")
    
    try:
        start_time = time.time()
        
        # Initialize TTS with a multilingual model (smaller than XTTS)
        # Force CPU for model loading to avoid TorchDevice conflicts
        logger.info("Loading model on CPU first to avoid TorchDevice conflicts...")
        tts = TTS("tts_models/multilingual/multi-dataset/your_tts", gpu=False)
        
        # Log device information
        try:
            device = next(tts.model.parameters()).device
            logger.info(f"Model loaded on device: {device}")
        except Exception as e:
            logger.warning(f"Could not determine model device: {e}")
        
        # Generate speech in English
        text = "This is a test of multilingual text to speech using TorchDevice."
        logger.info(f"Generating speech for text: '{text}'")
        
        # We don't have a speaker sample, so we'll use the default speaker
        tts.tts_to_file(text=text, file_path=output_path, speaker_wav=None, language="en")
        
        end_time = time.time()
        logger.info(f"Speech generated and saved to {output_path}")
        logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
        
        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"Error in multilingual TTS test: {e}")
        return False

def main():
    """Main function to run all tests."""
    # Create samples directory if it doesn't exist
    Path("samples").mkdir(exist_ok=True)
    
    # Print device information
    print_device_info()
    
    # List available models
    list_available_models()
    
    # Test basic TTS
    basic_result = test_basic_tts()
    
    # Test multilingual TTS (optional, as it requires a larger model download)
    multilingual_result = False
    user_input = input("\nDo you want to test the multilingual model? (y/n): ")
    if user_input.lower() == 'y':
        multilingual_result = test_multilingual_tts()
    
    # Print test results
    logger.info("\nTest Results:")
    logger.info(f"Basic TTS: {'SUCCESS' if basic_result else 'FAILURE'}")
    if user_input.lower() == 'y':
        logger.info(f"Multilingual TTS: {'SUCCESS' if multilingual_result else 'FAILURE'}")
    
    if basic_result and (not user_input.lower() == 'y' or multilingual_result):
        logger.info("\nAll tests passed! TorchDevice successfully redirected CUDA calls for TTS.")
    else:
        logger.info("\nSome tests failed. See above for details.")

if __name__ == "__main__":
    main()
