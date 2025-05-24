#!/usr/bin/env python
"""
Test nested function calls with TorchDevice.

This module tests the logging of nested function calls with TorchDevice,
ensuring that the correct caller information is captured.
"""
import logging
import unittest
from pathlib import Path

import torch
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture, LogCapture
from nested_tests.outer_module import outer_function, outer_wrapper

import TorchDevice  # Import TorchDevice to ensure CUDA redirection is set up

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Ensure this configuration is applied
)
logger = logging.getLogger(__name__)


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

    def run_device_operations(self) -> LogCapture:
        """
        Run device operations with nested function calls and return the captured log output.
        """
        # Call the nested functions defined in this file
        result_tensor1 = nested_function_level_1()
        tensor_sum1 = result_tensor1.sum().item()
        print(f"Sum of result tensor 1: {tensor_sum1}\n")
        
        # Call the nested functions from imported modules
        result_tensor2 = outer_function()
        tensor_sum2 = result_tensor2.item()  # outer_function returns a scalar
        print(f"Sum of result tensor 2: {tensor_sum2}\n")

        # Call the wrapper function for deeper nesting
        result_tensor3 = outer_wrapper()
        tensor_sum3 = result_tensor3.item()  # outer_wrapper returns a scalar
        print(f"Sum of result tensor 3: {tensor_sum3}\n")

        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        self.log_capture.log_stream.getvalue()

        return self.log_capture

    def test_nested_calls(self):
        """
        Test nested function calls by generating log output and performing a diff check.
        """
        # Run device operations and get the captured log
        self.log_capture = self.run_device_operations()
        
        # Perform the diff check
        diff_check(self.log_capture)


if __name__ == "__main__":
    unittest.main() 