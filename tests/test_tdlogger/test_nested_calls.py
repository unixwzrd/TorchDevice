#!/usr/bin/env python
"""
Test nested function calls with TorchDevice.

This module tests the logging of nested function calls with TorchDevice,
ensuring that the correct caller information is captured.
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


# Create some nested functions to test the logger's ability to track the call stack
def nested_function_level_1():
    """First level of nested function calls."""
    return nested_function_level_2()


def nested_function_level_2():
    """Second level of nested function calls."""
    return nested_function_level_3()


def nested_function_level_3():
    """
    Third level of nested function calls.
    
    This function creates a tensor and moves it to the GPU,
    which should trigger a log message with the correct caller info.
    """
    # Create a tensor
    tensor = torch.randn(2, 3)
    
    # Move the tensor to the GPU (will be redirected to MPS on Mac)
    result = tensor.cuda()
    
    return result


class TestNestedCalls(unittest.TestCase):
    """Test nested function calls with TorchDevice."""
    
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
        self.expected_output_file = Path(__file__).parent / "test_nested_calls_expected.log"
        
        # Define the temp output file path
        self.temp_output_file = Path(__file__).parent / "test_nested_calls_temp.log"
        
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

    def test_nested_calls(self):
        """
        Test nested function calls by generating log output and performing a diff check.
        """
        # Call the nested functions
        result_tensor = nested_function_level_1()
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Calculate the sum of the tensor for verification
        tensor_sum = result_tensor.sum().item()
        print(f"Sum of result tensor: {tensor_sum}")
        
        # Get the captured log output
        captured_log = self.log_stream.getvalue()
        
        # Perform the diff check
        diff_check(captured_log, self.expected_output_file)


if __name__ == "__main__":
    unittest.main() 