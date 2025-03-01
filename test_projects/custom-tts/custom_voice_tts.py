#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom Voice TTS with TorchDevice integration.
This script demonstrates using Coqui TTS with voice customization and TorchDevice.
"""

import os
import sys
import time
import torch
import argparse
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
    logger.info("Custom Voice TTS with TorchDevice")
    logger.info("============================================================")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Device count: {torch.cuda.device_count()}")
    logger.info(f"Current device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name()}")
    
    # Get the default device using TorchDevice
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Default device: {default_device}")
    logger.info("============================================================")

def generate_tts_with_custom_voice(text, speaker_wav, output_path, model_name="tts_models/multilingual/multi-dataset/your_tts", language="en"):
    """
    Generate speech with a custom voice using the specified model.
    
    Args:
        text (str): The text to convert to speech
        speaker_wav (str): Path to the speaker sample WAV file
        output_path (str): Path to save the generated speech
        model_name (str): Name of the TTS model to use
        language (str): Language code (e.g., "en", "fr", "es")
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Initializing TTS model: {model_name}")
        start_time = time.time()
        
        # Initialize TTS with the specified model
        # Force CPU for model loading to avoid TorchDevice conflicts
        logger.info("Loading model on CPU first to avoid TorchDevice conflicts...")
        tts = TTS(model_name, gpu=False)
        
        # Log device information
        try:
            device = next(tts.model.parameters()).device
            logger.info(f"Model loaded on device: {device}")
        except Exception as e:
            logger.warning(f"Could not determine model device: {e}")
        
        # Generate speech with the custom voice
        logger.info(f"Generating speech for text: '{text}'")
        if speaker_wav:
            logger.info(f"Using speaker sample: {speaker_wav}")
        else:
            logger.info("Using default voice (no speaker sample provided)")
        
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=speaker_wav,
            language=language
        )
        
        end_time = time.time()
        logger.info(f"Speech generated and saved to {output_path}")
        logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
        
        return True
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return False

def list_voice_cloning_models():
    """List models that support voice cloning."""
    logger.info("Models that support voice cloning:")
    
    try:
        # Get all models
        tts_instance = TTS()
        all_models = tts_instance.list_models()
        
        # Known models that support voice cloning
        voice_cloning_keywords = ["your_tts", "xtts", "voice_conversion"]
        voice_cloning_models = []
        
        for model in all_models:
            if any(keyword in model for keyword in voice_cloning_keywords):
                voice_cloning_models.append(model)
                logger.info(f"  - {model}")
        
        if not voice_cloning_models:
            logger.info("  No voice cloning models found.")
        
        return voice_cloning_models
    except Exception as e:
        logger.error(f"Error listing voice cloning models: {e}")
        logger.info("Continuing with default models...")
        # Return some known voice cloning models as fallback
        return ["tts_models/multilingual/multi-dataset/your_tts"]

def main():
    """Main function to parse arguments and run TTS."""
    parser = argparse.ArgumentParser(description="Generate speech with a custom voice using TTS and TorchDevice")
    parser.add_argument("--text", type=str, default="Hello, this is a test of voice cloning with TorchDevice.",
                        help="Text to convert to speech")
    parser.add_argument("--speaker", type=str, required=False,
                        help="Path to the speaker sample WAV file")
    parser.add_argument("--output", type=str, default="samples/custom_voice_output.wav",
                        help="Path to save the generated speech")
    parser.add_argument("--model", type=str, default="tts_models/multilingual/multi-dataset/your_tts",
                        help="Name of the TTS model to use")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code (e.g., 'en', 'fr', 'es')")
    parser.add_argument("--list-models", action="store_true",
                        help="List models that support voice cloning")
    
    args = parser.parse_args()
    
    # Create samples directory if it doesn't exist
    Path(os.path.dirname(args.output)).mkdir(exist_ok=True, parents=True)
    
    # Print device information
    print_device_info()
    
    # List voice cloning models if requested
    if args.list_models:
        list_voice_cloning_models()
        return
    
    # Check if speaker sample is provided for voice cloning models
    if "your_tts" in args.model or "xtts" in args.model:
        if not args.speaker:
            logger.warning("No speaker sample provided. For voice cloning models, this will use the default voice.")
            user_input = input("Do you want to continue with the default voice? (y/n): ")
            if user_input.lower() != 'y':
                logger.info("Exiting...")
                return
    
    # Generate speech with the custom voice
    success = generate_tts_with_custom_voice(
        text=args.text,
        speaker_wav=args.speaker,
        output_path=args.output,
        model_name=args.model,
        language=args.language
    )
    
    if success:
        logger.info("Speech generation completed successfully!")
    else:
        logger.error("Speech generation failed.")

if __name__ == "__main__":
    main()
