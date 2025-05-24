#!/usr/bin/env python
"""
Demo: Small Transformers Model with TorchDevice
This script demonstrates running a small Hugging Face Transformers model using TorchDevice for device-agnostic inference.
"""

import os
import sys
import time
import logging

# Import TorchDevice first to patch PyTorch
try:
    import TorchDevice
except ImportError:
    print("TorchDevice not found. Please install it before running this demo.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transformers_demo")

# Import torch and transformers
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except ImportError as e:
    logger.error(f"Required package not found: {e}")
    sys.exit(1)

def main():
    # Print device info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"MPS available: {getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Default device: {getattr(TorchDevice, 'TorchDevice', None) and TorchDevice.TorchDevice.get_default_device()}")

    # Choose model
    model_name = os.environ.get("TRANSFORMERS_MODEL", "hf-internal-testing/tiny-random-bert")

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loaded tokenizer: {tokenizer}")
        model = AutoModelForMaskedLM.from_pretrained(model_name)
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

    # Prepare input for masked language modeling
    input_text = "The quick brown [MASK] jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Run inference
    logger.info("Running inference...")
    with torch.no_grad():
        start_time = time.time()
        outputs = model(**inputs)
        elapsed = time.time() - start_time
    logger.info(f"Inference completed in {elapsed*1000:.2f} ms")

    # Get the predicted token for [MASK]
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
    predicted_token_id = outputs.logits[0, mask_token_index].argmax(dim=-1).item()
    predicted_token = tokenizer.decode([predicted_token_id])
    logger.info(f"Input: {input_text}")
    logger.info(f"Predicted token for [MASK]: {predicted_token}")

if __name__ == "__main__":
    main() 