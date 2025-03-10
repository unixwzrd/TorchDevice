#!/usr/bin/env python
"""
Test basic logging functionality with TorchDevice.

This module tests the basic logging functionality of the TDLogger module,
including capturing and verifying log messages.
"""
import logging
import unittest
from pathlib import Path

import torch
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture

import TorchDevice  # Import TorchDevice to ensure CUDA redirection is set up


class TestTDLogger(unittest.TestCase):

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
        
        self.captured_log = self.log_capture.log_stream.getvalue()

        # Get the captured log output
        
        return self.log_capture

    def test_vector_operations(self):
        """
        Test vector operations by generating log output and performing a diff check.
        """
        # Run the device operations and get the captured log
        self.log_capture = self.run_device_operations()
        
        # Perform the diff check
        diff_check(self.log_capture)


if __name__ == "__main__":
    unittest.main()