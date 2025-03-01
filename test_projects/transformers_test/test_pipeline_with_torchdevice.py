#!/usr/bin/env python
"""
Test script for running transformers pipelines with TorchDevice.

This script tests various transformers pipelines with TorchDevice to ensure
that CUDA code works correctly on MPS devices.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import TorchDevice first to patch PyTorch
import TorchDevice

# Now import torch and check available devices
import torch
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"MPS available: {torch.backends.mps.is_available()}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")  # Should return True on MPS devices thanks to TorchDevice
logger.info(f"Default device: {TorchDevice.TorchDevice.get_default_device()}")

# Apply our torch._dynamo patch before importing transformers
from torch_dynamo_patch import apply_torch_dynamo_patch
apply_torch_dynamo_patch()

# Import transformers
from transformers import pipeline, AutoTokenizer, AutoModel

def test_pipeline(task: str, model: str = None, device: str = "cuda", **kwargs) -> Dict[str, Any]:
    """
    Test a transformers pipeline with TorchDevice.
    
    Args:
        task: The pipeline task to test
        model: The model to use (or None for default)
        device: The device to use ('cuda', 'mps', or 'cpu')
        **kwargs: Additional arguments to pass to the pipeline
        
    Returns:
        Dict with test results
    """
    logger.info(f"Testing {task} pipeline with device={device}")
    
    # Create the pipeline
    start_time = time.time()
    pipe = pipeline(task, model=model, device=device, **kwargs)
    init_time = time.time() - start_time
    logger.info(f"Pipeline initialized in {init_time:.2f} seconds")
    
    # Get appropriate test input based on the task
    test_input = get_test_input(task)
    logger.info(f"Running inference with input: {test_input}")
    
    # Run inference
    start_time = time.time()
    result = pipe(test_input)
    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.2f} seconds")
    
    # Log device information
    if hasattr(pipe, "device"):
        logger.info(f"Pipeline device: {pipe.device}")
    if hasattr(pipe, "model") and hasattr(pipe.model, "device"):
        logger.info(f"Model device: {pipe.model.device}")
    
    # Return results
    return {
        "task": task,
        "device": device,
        "init_time": init_time,
        "inference_time": inference_time,
        "result": result
    }

def get_test_input(task: str) -> Union[str, List[str], Dict]:
    """Get appropriate test input based on the task."""
    inputs = {
        "text-classification": "I love this movie!",
        "token-classification": "My name is John and I live in New York",
        "question-answering": {"question": "What is the capital of France?", "context": "Paris is the capital of France."},
        "summarization": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930.",
        "translation": "Hello, how are you?",
        "text-generation": "Once upon a time,",
        "fill-mask": "The capital of France is [MASK].",
        "feature-extraction": "Hello, my name is John",
        "image-classification": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        "object-detection": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        "image-segmentation": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        "text-to-image": "A photo of a cat",
        "image-to-text": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        "automatic-speech-recognition": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        "audio-classification": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        "text-to-audio": "Hello, my name is John",
        "depth-estimation": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    }
    
    return inputs.get(task, "Hello world")

def test_all_pipelines(device="cuda"):
    """Test all supported pipelines."""
    results = {}
    
    # Text pipelines
    text_tasks = [
        "text-classification",
        "token-classification",
        "question-answering",
        "summarization",
        "text-generation",
        "fill-mask",
        "feature-extraction"
    ]
    
    # Test text pipelines with small models
    for task in text_tasks:
        try:
            # Use small models for faster testing
            model = None
            if task == "text-classification":
                model = "distilbert-base-uncased-finetuned-sst-2-english"
            elif task == "token-classification":
                model = "hf-internal-testing/tiny-bert-for-token-classification"
            elif task == "question-answering":
                model = "hf-internal-testing/tiny-bert-for-question-answering"
            elif task == "summarization":
                model = "sshleifer/tiny-mbart"
            elif task == "text-generation":
                model = "sshleifer/tiny-gpt2"
            elif task == "fill-mask":
                model = "hf-internal-testing/tiny-bert-for-token-classification"
            elif task == "feature-extraction":
                model = "hf-internal-testing/tiny-bert-for-token-classification"
                
            results[task] = test_pipeline(task, model=model, device=device)
            logger.info(f"✅ {task} pipeline test passed")
        except Exception as e:
            logger.error(f"❌ {task} pipeline test failed: {str(e)}")
            results[task] = {"error": str(e)}
    
    return results

def test_basic_model_operations():
    """Test basic model operations with TorchDevice."""
    logger.info("Testing basic model operations with TorchDevice")
    
    # Use a tiny model for testing
    model_name = "hf-internal-testing/tiny-random-bert"
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        logger.info(f"Model loaded: {model_name}")
        logger.info(f"Initial model device: {model.device}")
        
        # Move to CUDA (should be redirected to MPS on Apple Silicon)
        device = torch.device("cuda")
        logger.info(f"Requested device: {device}")
        logger.info(f"Actual device type: {device.type}")
        
        model = model.to(device)
        logger.info(f"Model device after moving: {model.device}")
        
        # Test inference
        inputs = tokenizer("Hello world", return_tensors="pt").to(device)
        logger.info(f"Input tensor device: {inputs.input_ids.device}")
        
        with torch.no_grad():
            start_time = time.time()
            outputs = model(**inputs)
            inference_time = time.time() - start_time
        
        logger.info(f"Output tensor device: {outputs.last_hidden_state.device}")
        logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
        logger.info(f"Inference time: {inference_time * 1000:.2f} ms")
        
        # Test on CPU for comparison
        model = model.to("cpu")
        inputs = tokenizer("Hello world", return_tensors="pt")
        
        with torch.no_grad():
            start_time = time.time()
            outputs = model(**inputs)
            cpu_inference_time = time.time() - start_time
        
        logger.info(f"CPU inference time: {cpu_inference_time * 1000:.2f} ms")
        logger.info("✅ Basic model operations test passed")
        
        return {
            "success": True,
            "inference_time_ms": inference_time * 1000,
            "cpu_inference_time_ms": cpu_inference_time * 1000,
            "speedup": cpu_inference_time / inference_time if inference_time > 0 else 0
        }
    
    except Exception as e:
        logger.error(f"❌ Basic model operations test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Test basic model operations
    basic_results = test_basic_model_operations()
    
    # Test a subset of pipelines
    if len(sys.argv) > 1:
        # Test specific pipeline if provided as argument
        task = sys.argv[1]
        test_pipeline(task, device="cuda")
    else:
        # Test text classification as a representative pipeline
        test_pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device="cuda")
        
        # Uncomment to test all pipelines (can be slow)
        # test_all_pipelines(device="cuda")
