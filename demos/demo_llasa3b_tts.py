#!/usr/bin/env python
"""
Demo: LLASA3B TTS with TorchDevice (Unpatched, Standard Usage)
This script demonstrates running a TTS model using TorchDevice for device-agnostic inference.
No device-specific patching or workarounds are used—just import TorchDevice and run standard code.
"""

import logging
import os
import sys
import time

# Import TorchDevice first to patch PyTorch
import TorchDevice  # Required for side effects

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llasa3b_demo")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    logger.error(f"Required package not found: {e}")
    sys.exit(1)

# Try to import soundfile for audio output
try:
    import soundfile  # noqa: F401  # Not used in this demo, but shown for completeness
    SOUND_OUTPUT = True
except ImportError:
    logger.warning("soundfile not found, audio output will be skipped.")
    SOUND_OUTPUT = False

def main():
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"MPS available: {getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    # Use environment variable or default to local Llasa-3B model directory
    model_name = os.environ.get(
        "LLASA3B_MODEL",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_projects", "speech", "Llasa-3B"))
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info(f"Loaded model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        sys.exit(1)

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else "cpu"
        )
    )
    logger.info(f"Using device: {device}")
    model = model.to(device)

    # Prepare input
    input_text = "Hello, this is a test of the LLASA3B (or fallback) TTS system with TorchDevice integration."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Run inference
    logger.info("Running inference...")
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_length=64,
            do_sample=True,
            top_p=0.95,
            temperature=0.8
        )
        elapsed = time.time() - start_time
    logger.info(f"Inference completed in {elapsed*1000:.2f} ms")

    # Decode output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated output: {output_text}")

    # No audio synthesis in this minimal demo
    logger.info("No audio synthesis performed in this demo. Output is text only.")


if __name__ == "__main__":
    main() 