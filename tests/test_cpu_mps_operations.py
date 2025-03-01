#!/usr/bin/env python
import unittest
import torch
import numpy as np
import sys
import os
import logging
import TorchDevice  # Ensure this module is imported to apply patches

# Add the current directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_utils import PrefixedTestCase  # Import our custom TestCase

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Ensure this configuration is applied
)
logger = logging.getLogger(__name__)

# Prevent duplicate logging
logging.getLogger("TorchDevice").propagate = False

class TestCPUOperations(PrefixedTestCase):
    """
    Test case for CPU operations to ensure they work correctly with TorchDevice.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Always use CPU for these tests
        self.device = torch.device('cpu')
        self.info("Using device: %s", self.device)
    
    def test_cpu_tensor_creation(self):
        """Test creating tensors explicitly on CPU."""
        self.info("Testing CPU tensor creation")
        
        # Create tensors with different methods
        tensor1 = torch.randn(10, device='cpu')
        tensor2 = torch.zeros(10, device='cpu')
        tensor3 = torch.ones(10, device='cpu')
        tensor4 = torch.tensor([1, 2, 3], device='cpu')
        
        # Verify they're all on CPU
        self.assertEqual(tensor1.device.type, 'cpu')
        self.assertEqual(tensor2.device.type, 'cpu')
        self.assertEqual(tensor3.device.type, 'cpu')
        self.assertEqual(tensor4.device.type, 'cpu')
        
        self.info("CPU tensor creation tests passed")
    
    def test_cpu_tensor_operations(self):
        """Test operations on CPU tensors."""
        self.info("Testing CPU tensor operations")
        
        # Create tensors
        a = torch.randn(10, device='cpu')
        b = torch.randn(10, device='cpu')
        
        # Test basic operations
        c = a + b
        d = a * b
        e = torch.matmul(a, b)
        
        # Verify results are on CPU
        self.assertEqual(c.device.type, 'cpu')
        self.assertEqual(d.device.type, 'cpu')
        self.assertEqual(e.device.type, 'cpu')
        
        self.info("CPU tensor operations tests passed")
    
    def test_cpu_nn_operations(self):
        """Test neural network operations on CPU."""
        self.info("Testing CPU neural network operations")
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).to('cpu')
        
        # Create input
        x = torch.randn(3, 10, device='cpu')
        
        # Forward pass
        output = model(x)
        
        # Verify output is on CPU
        self.assertEqual(output.device.type, 'cpu')
        
        # Check model parameters are on CPU
        for param in model.parameters():
            self.assertEqual(param.device.type, 'cpu')
        
        self.info("CPU neural network operations tests passed")


class TestMPSOperations(PrefixedTestCase):
    """
    Test case for MPS operations to ensure they work correctly with TorchDevice.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Check if MPS is available
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        self.has_cuda = torch.cuda.is_available()
        
        # Skip tests if neither MPS nor CUDA is available
        if not self.has_mps and not self.has_cuda:
            self.skipTest("Neither MPS nor CUDA is available")
        
        # Use MPS device
        self.device = torch.device('mps')
        self.info("Using device: %s", self.device)
    
    def test_mps_tensor_creation(self):
        """Test creating tensors explicitly on MPS."""
        self.info("Testing MPS tensor creation")
        
        # Create tensors with different methods
        tensor1 = torch.randn(10, device='mps')
        tensor2 = torch.zeros(10, device='mps')
        tensor3 = torch.ones(10, device='mps')
        tensor4 = torch.tensor([1, 2, 3], device='mps')
        
        # Verify they're on the expected device
        expected_type = 'mps' if self.has_mps else 'cuda' if self.has_cuda else 'cpu'
        self.assertEqual(tensor1.device.type, expected_type)
        self.assertEqual(tensor2.device.type, expected_type)
        self.assertEqual(tensor3.device.type, expected_type)
        self.assertEqual(tensor4.device.type, expected_type)
        
        self.info("MPS tensor creation tests passed")
    
    def test_mps_tensor_operations(self):
        """Test operations on MPS tensors."""
        self.info("Testing MPS tensor operations")
        
        # Create tensors
        a = torch.randn(10, device='mps')
        b = torch.randn(10, device='mps')
        
        # Test basic operations
        c = a + b
        d = a * b
        e = torch.matmul(a, b)
        
        # Verify results are on the expected device
        expected_type = 'mps' if self.has_mps else 'cuda' if self.has_cuda else 'cpu'
        self.assertEqual(c.device.type, expected_type)
        self.assertEqual(d.device.type, expected_type)
        self.assertEqual(e.device.type, expected_type)
        
        self.info("MPS tensor operations tests passed")
    
    def test_mps_nn_operations(self):
        """Test neural network operations on MPS."""
        self.info("Testing MPS neural network operations")
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).to('mps')
        
        # Create input
        x = torch.randn(3, 10, device='mps')
        
        # Forward pass
        output = model(x)
        
        # Verify output is on the expected device
        expected_type = 'mps' if self.has_mps else 'cuda' if self.has_cuda else 'cpu'
        self.assertEqual(output.device.type, expected_type)
        
        # Check model parameters are on the expected device
        for param in model.parameters():
            self.assertEqual(param.device.type, expected_type)
        
        self.info("MPS neural network operations tests passed")
    
    def test_cpu_to_mps_conversion(self):
        """Test converting tensors from CPU to MPS."""
        self.info("Testing CPU to MPS conversion")
        
        # Create a CPU tensor
        cpu_tensor = torch.randn(10, device='cpu')
        self.assertEqual(cpu_tensor.device.type, 'cpu')
        
        # Convert to MPS
        mps_tensor = cpu_tensor.to('mps')
        
        # Verify it's on the expected device
        expected_type = 'mps' if self.has_mps else 'cuda' if self.has_cuda else 'cpu'
        self.assertEqual(mps_tensor.device.type, expected_type)
        
        # Convert back to CPU
        cpu_tensor_again = mps_tensor.to('cpu')
        self.assertEqual(cpu_tensor_again.device.type, 'cpu')
        
        self.info("CPU to MPS conversion tests passed")
    
    def test_mps_device_properties(self):
        """Test MPS device properties."""
        self.info("Testing MPS device properties")
        
        # Get MPS device
        device = torch.device('mps:0')
        
        # Check device type
        expected_type = 'mps' if self.has_mps else 'cuda' if self.has_cuda else 'cpu'
        self.assertEqual(device.type, expected_type)
        
        # If it's redirected to MPS, check index
        if device.type == 'mps':
            self.assertEqual(device.index, 0)
        
        self.info("MPS device properties tests passed")


if __name__ == '__main__':
    unittest.main()
