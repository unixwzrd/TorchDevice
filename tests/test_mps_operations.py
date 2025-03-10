#!/usr/bin/env python
"""
Test file for MPS device operations with TorchDevice.
This ensures that all operations work correctly on MPS.
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


class TestMPSOperations(PrefixedTestCase):
    """
    Test case for MPS operations to ensure they work correctly with TorchDevice.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Call the parent setUp method to set up logging
        super().setUp()
        
        # Check if MPS is available
        self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Set the device to MPS if available, otherwise CPU
        if self.mps_available:
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')  # Fallback to CPU
            
        self.info(f"Using device: {self.device}")
        
        # Skip tests if MPS is not available
        if not self.mps_available:
            self.skipTest("MPS is not available on this system")
        
        # Set up log capture
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)
    
    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        teardown_log_capture(self.log_capture)
        super().tearDown()
    
    def test_mps_tensor_creation(self):
        """Test creating tensors explicitly on MPS."""
        self.info("Testing MPS tensor creation")
        
        # Create tensors on MPS
        tensor1 = torch.randn(2, 3, device=self.device)
        tensor2 = torch.zeros(3, 4, device=self.device)
        tensor3 = torch.ones(2, 2, device=self.device)
        tensor4 = torch.tensor([1, 2, 3], device=self.device)
        
        # Check device type
        expected_type = 'mps' if self.mps_available else 'cpu'
        self.assertEqual(tensor1.device.type, expected_type)
        self.assertEqual(tensor2.device.type, expected_type)
        self.assertEqual(tensor3.device.type, expected_type)
        self.assertEqual(tensor4.device.type, expected_type)
        
        self.info("MPS tensor creation tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_mps_tensor_operations(self):
        """Test operations on MPS tensors."""
        self.info("Testing MPS tensor operations")
        
        # Create tensors on MPS
        a = torch.randn(2, 3, device=self.device)
        b = torch.randn(3, 2, device=self.device)
        
        # Perform operations
        c = torch.matmul(a, b)
        d = torch.nn.functional.relu(c)
        
        # Check device type
        expected_type = 'mps' if self.mps_available else 'cpu'
        self.assertEqual(c.device.type, expected_type)
        self.assertEqual(d.device.type, expected_type)
        
        self.info("MPS tensor operations tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_mps_nn_operations(self):
        """Test neural network operations on MPS."""
        self.info("Testing MPS neural network operations")
        
        # Create a simple neural network and move it to MPS
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).to(self.device)
        
        # Create input data on MPS
        x = torch.randn(3, 10, device=self.device)
        
        # Forward pass
        output = model(x)
        
        # Check device type
        expected_type = 'mps' if self.mps_available else 'cpu'
        self.assertEqual(output.device.type, expected_type)
        
        # Check model parameters are on MPS
        for param in model.parameters():
            self.assertEqual(param.device.type, expected_type)
        
        self.info("MPS neural network operations tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_cpu_to_mps_conversion(self):
        """Test converting tensors from CPU to MPS."""
        self.info("Testing CPU to MPS conversion")
        
        # Create tensor on CPU
        cpu_tensor = torch.randn(2, 3)
        self.assertEqual(cpu_tensor.device.type, 'cpu')
        
        # Convert to MPS
        mps_tensor = cpu_tensor.to(self.device)
        
        # Check device type
        expected_type = 'mps' if self.mps_available else 'cpu'
        self.assertEqual(mps_tensor.device.type, expected_type)
        
        # Verify values are the same
        cpu_tensor_again = mps_tensor.cpu()
        self.assertTrue(torch.allclose(cpu_tensor, cpu_tensor_again))
        
        self.info("CPU to MPS conversion tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_mps_device_properties(self):
        """Test MPS device properties."""
        self.info("Testing MPS device properties")
        
        # Get MPS device
        device = torch.device('mps:0')
        
        # Check device type
        expected_type = 'mps' if self.mps_available else 'cpu'
        self.assertEqual(device.type, expected_type)
        
        # Check device index
        if self.mps_available:
            self.assertEqual(device.index, 0)
        
        self.info("MPS device properties tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)


if __name__ == '__main__':
    unittest.main() 