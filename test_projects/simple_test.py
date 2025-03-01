#!/usr/bin/env python
"""
Simple test for TorchDevice functionality.
This script tests basic tensor operations with TorchDevice.
"""

import os
import sys
import time
import logging

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
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"Default device: {TorchDevice.TorchDevice.get_default_device()}")

def test_basic_tensor_operations():
    """Test basic tensor operations with TorchDevice."""
    logger.info("Testing basic tensor operations with TorchDevice")
    
    try:
        # Create a tensor on CPU
        x_cpu = torch.randn(3, 3)
        logger.info(f"CPU tensor device: {x_cpu.device}")
        
        # Move to CUDA (should be redirected to MPS on Apple Silicon)
        device = torch.device("cuda")
        logger.info(f"Requested device: {device}")
        logger.info(f"Actual device type: {device.type}")
        
        x_gpu = x_cpu.to(device)
        logger.info(f"GPU tensor device: {x_gpu.device}")
        
        # Test basic operations
        y_gpu = x_gpu + x_gpu
        logger.info(f"Addition result device: {y_gpu.device}")
        
        z_gpu = torch.matmul(x_gpu, y_gpu)
        logger.info(f"Matrix multiplication result device: {z_gpu.device}")
        
        # Test stream operations
        s = torch.cuda.Stream()
        logger.info(f"Stream created: {s}")
        
        with torch.cuda.stream(s):
            w_gpu = z_gpu * 2
        
        logger.info(f"Operation with stream result device: {w_gpu.device}")
        
        # Test event operations
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(100):
            w_gpu = z_gpu * 2
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        logger.info(f"Event timing: {elapsed_time} ms")
        
        logger.info("✅ Basic tensor operations test passed")
        return True
    
    except Exception as e:
        logger.error(f"❌ Basic tensor operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_tensor_operations()
