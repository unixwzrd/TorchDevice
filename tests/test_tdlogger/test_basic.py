#!/usr/bin/env python
"""
Test basic logging functionality with TorchDevice.

This module tests the basic logging functionality of the TDLogger module,
including capturing and verifying log messages.
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


class TestTDLogger(unittest.TestCase):
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
        self.expected_output_file = Path(__file__).parent / "test_vector_operations_expected.log"
        
        # Define the temp output file path
        self.temp_output_file = Path(__file__).parent / "test_vector_operations_temp.log"
        
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

    def run_device_operations(self) -> str:
        """
        Run some basic device operations and return the captured log output.
        """
        # Create a tensor on the CPU
        x = torch.tensor([1.0, 2.0, 3.0])
        
        # Move the tensor to the GPU (will be redirected to MPS on Mac)
        _ = x.cuda()
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Get the captured log output
        captured_log = self.log_stream.getvalue()
        
        return captured_log

    def test_vector_operations(self):
        """
        Test vector operations by generating log output and performing a diff check.
        """
        # Run the device operations and get the captured log
        captured_log = self.run_device_operations()
        
        # Perform the diff check
        diff_check(captured_log, self.expected_output_file)


if __name__ == "__main__":

    unittest.main()