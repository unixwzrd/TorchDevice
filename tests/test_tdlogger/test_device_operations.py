"""
Test device operations with TorchDevice.

This module tests various device operations with TorchDevice,
ensuring that the correct log messages are generated.
"""

import os
import sys
import unittest
from pathlib import Path

import torch
import TorchDevice  # Import TorchDevice to ensure CUDA redirection is set up

# Add the current directory to the path so we can import test_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from test_utils import diff_check, setup_log_capture, teardown_log_capture


class TestDeviceOperations(unittest.TestCase):
    """Test device operations with TorchDevice."""
    
    def setUp(self):
        """Set up logger configuration for this test."""
        # Print a header for the test
        print("\n" + "=" * 80)
        print(f"Starting test: {self._testMethodName}")
        print("=" * 80)
        
        # Set up the logger
        result = setup_log_capture()
        self.logger = result[0]
        self.log_stream = result[1]
        self.log_handler = result[2]
        self.console_handler = result[3]
        self.original_handlers = result[4]
        self.original_level = result[5]
        
        # Define the expected output file path
        self.expected_output_file = Path(__file__).parent / f"{self._testMethodName}_expected.log"
        
        # Define the temp output file path
        self.temp_output_file = Path(__file__).parent / f"{self._testMethodName}_temp.log"
        
        # Clear the temp file if it exists
        if self.temp_output_file.exists():
            self.temp_output_file.unlink()

    def tearDown(self):
        """Clean up after the test."""
        # Remove our handlers and restore original configuration
        teardown_log_capture(
            self.logger, 
            self.original_handlers, 
            self.original_level,
            [self.log_handler, self.console_handler]
        )
        
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
        captured_log = self.log_stream.getvalue()
        
        # Perform the diff check
        diff_check(captured_log, self.expected_output_file)

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
        captured_log = self.log_stream.getvalue()
        
        # Perform the diff check
        diff_check(captured_log, self.expected_output_file)

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
        captured_log = self.log_stream.getvalue()
        
        # Perform the diff check
        diff_check(captured_log, self.expected_output_file) 