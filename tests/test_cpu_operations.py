#!/usr/bin/env python
"""
Test file for CPU device operations with TorchDevice.
This ensures that all operations work correctly on CPU with explicit CPU override.
"""
import logging
import unittest
import torch
from pathlib import Path

# Import from common test utilities
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture
from common.test_utils import PrefixedTestCase

# Import TorchDevice to ensure CUDA redirection is set up
import TorchDevice

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestCPUOperations(PrefixedTestCase):
    """
    Test case for CPU operations to ensure they work correctly with TorchDevice.
    """
    
    def setUp(self):
        """Set up test environment with CPU override."""
        # Call the parent setUp method to set up logging
        super().setUp()
        
        # Explicitly override to CPU using the special device notation
        self.device = torch.device('cpu:-1')
        self.info(f"Using device: {self.device}")
        
        # Set up log capture
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)
    
    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        teardown_log_capture(self.log_capture)
        super().tearDown()
    
    def test_cpu_tensor_creation(self):
        """Test creating tensors explicitly on CPU."""
        self.info("Testing CPU tensor creation")
        
        # Create tensors on CPU
        cpu_tensor1 = torch.randn(2, 3, device='cpu')
        cpu_tensor2 = torch.zeros(3, 4, device='cpu')
        
        # Verify tensors are on CPU
        self.assertEqual(cpu_tensor1.device.type, 'cpu')
        self.assertEqual(cpu_tensor2.device.type, 'cpu')
        
        self.info("CPU tensor creation tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_cpu_tensor_operations(self):
        """Test operations on CPU tensors."""
        self.info("Testing CPU tensor operations")
        
        # Create tensors on CPU
        a = torch.randn(2, 3, device='cpu')
        b = torch.randn(3, 2, device='cpu')
        
        # Perform operations
        c = torch.matmul(a, b)
        d = torch.nn.functional.relu(c)
        
        # Verify tensors are on CPU
        self.assertEqual(c.device.type, 'cpu')
        self.assertEqual(d.device.type, 'cpu')
        
        self.info("CPU tensor operations tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_cpu_nn_operations(self):
        """Test neural network operations on CPU."""
        self.info("Testing CPU neural network operations")
        
        # Create a simple neural network
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).to('cpu')
        
        # Create input data
        x = torch.randn(3, 10, device='cpu')
        
        # Forward pass
        output = model(x)
        
        # Verify output is on CPU
        self.assertEqual(output.device.type, 'cpu')
        
        # Check model parameters are on CPU
        for param in model.parameters():
            self.assertEqual(param.device.type, 'cpu')
        
        self.info("CPU neural network operations tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)


if __name__ == '__main__':
    unittest.main() 