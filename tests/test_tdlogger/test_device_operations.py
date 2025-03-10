"""
Test device operations with TorchDevice.

This module tests various device operations with TorchDevice,
ensuring that the correct log messages are generated.
"""
import logging
import unittest
from pathlib import Path

import torch
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture

import TorchDevice  # Import TorchDevice to ensure CUDA redirection is set up

class TestDeviceOperations(unittest.TestCase):
    """Test device operations with TorchDevice."""
    
    def setUp(self):
        """Set up logger capture for this test."""
        print("\n" + "=" * 80)
        print(f"Starting test: {self._testMethodName}")
        print("=" * 80)
        
        # Set up log capture with a unique test name.
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)        


    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        teardown_log_capture(self.log_capture)
        
        # Print a footer for the test
        print("\n" + "=" * 80)
        print(f"Finished test: {self._testMethodName}")
        print("=" * 80)
        

    def test_tensor_creation(self):
        """
        Test tensor creation operations by generating log output and performing a diff check.
        """
        # Create a tensor on the CPU
        _ = torch.tensor([1.0, 2.0, 3.0])
        
        # Create a tensor on the GPU (will be redirected to MPS on Mac)
        _ = torch.tensor([4.0, 5.0, 6.0], device='cuda')
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Get the captured log output
        self.log_capture.log_stream.getvalue()
        
        # Perform the diff check
        diff_check(self.log_capture)

    def test_tensor_movement(self):
        """
        Test tensor movement operations by generating log output and performing a diff check.
        """
        # Create a tensor on the CPU
        x_cpu = torch.tensor([1.0, 2.0, 3.0])
        
        # Move the tensor to the GPU (will be redirected to MPS on Mac)
        x_gpu = x_cpu.cuda()
        
        # Move the tensor back to the CPU
        _ = x_gpu.cpu()
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Get the captured log output
        self.log_capture.log_stream.getvalue()
        
        # Perform the diff check
        diff_check(self.log_capture)

    def test_cuda_functions(self):
        """
        Test CUDA functions by generating log output and performing a diff check.
        """
        # Check CUDA availability
        _ = torch.cuda.is_available()
        
        # Get device count
        _ = torch.cuda.device_count()
        
        # Get current device
        _ = torch.cuda.current_device()
        
        # Empty cache
        torch.cuda.empty_cache()
        
        # Synchronize
        torch.cuda.synchronize()
        
        # Get the captured log output
        self.log_capture.log_stream.getvalue()
        
        # Perform the diff check
        diff_check(self.log_capture) 


if __name__ == '__main__':
    unittest.main()
