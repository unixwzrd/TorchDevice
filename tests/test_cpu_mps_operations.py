#!/usr/bin/env python
import logging
import unittest
from pathlib import Path

import torch
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture
from common.test_utils import PrefixedTestCase

import TorchDevice  # Import TorchDevice to ensure CUDA redirection is set up
# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Ensure this configuration is applied
)
logger = logging.getLogger(__name__)

class TestCPUOperations(PrefixedTestCase):
    """
    Test case for CPU operations to ensure they work correctly with TorchDevice.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Call the parent setUp method to set up logging
        super().setUp()
        
        print("\n" + "=" * 80)
        print(f"Starting test: {self._testMethodName}")
        print("=" * 80)
        
        # Set up log capture with a unique test name.
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)   

        # Set device to CPU for these tests
        self.device = torch.device('cpu')
        self.info("Using device: %s", self.device)

    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        teardown_log_capture(self.log_capture)
        
        # Print a footer for the test
        print("\n" + "=" * 80)
        print(f"Finished test: {self._testMethodName}")
        print("=" * 80)


    def test_cpu_tensor_creation(self):
        """Test creating tensors explicitly on CPU."""
        self.info("Testing CPU tensor creation")
        
        # Create a tensor on CPU
        tensor1 = torch.tensor([1.0, 2.0, 3.0], device='cpu')
        tensor2 = torch.ones((2, 3), device='cpu')
        
        # Verify they're on CPU
        self.assertEqual(tensor1.device.type, 'cpu')
        self.assertEqual(tensor2.device.type, 'cpu')
        
        self.info("CPU tensor creation tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)
    
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

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)
    
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

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)


class TestMPSOperations(PrefixedTestCase):
    """
    Test case for MPS operations to ensure they work correctly with TorchDevice.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Call the parent setUp method to set up logging
        super().setUp()
        
        # Determine the available hardware - use MPS if available
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Skip tests if MPS is not available
        if not self.has_mps:
            self.skipTest("MPS is not available on this machine")
            
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
        expected_type = 'mps' if self.has_mps else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(tensor1.device.type, expected_type)
        self.assertEqual(tensor2.device.type, expected_type)
        self.assertEqual(tensor3.device.type, expected_type)
        self.assertEqual(tensor4.device.type, expected_type)
        
        self.info("MPS tensor creation tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

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
        expected_type = 'mps' if self.has_mps else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(c.device.type, expected_type)
        self.assertEqual(d.device.type, expected_type)
        self.assertEqual(e.device.type, expected_type)
        
        self.info("MPS tensor operations tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

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
        expected_type = 'mps' if self.has_mps else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(output.device.type, expected_type)
        
        # Check model parameters are on the expected device
        for param in model.parameters():
            self.assertEqual(param.device.type, expected_type)
        
        self.info("MPS neural network operations tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_cpu_to_mps_conversion(self):
        """Test converting tensors from CPU to MPS."""
        self.info("Testing CPU to MPS conversion")
        
        # Create a CPU tensor
        cpu_tensor = torch.randn(10, device='cpu')
        self.assertEqual(cpu_tensor.device.type, 'cpu')
        
        # Convert to MPS
        mps_tensor = cpu_tensor.to('mps')
        
        # Verify it's on the expected device
        expected_type = 'mps' if self.has_mps else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(mps_tensor.device.type, expected_type)
        
        # Convert back to CPU
        cpu_tensor_again = mps_tensor.to('cpu')
        self.assertEqual(cpu_tensor_again.device.type, 'cpu')
        
        self.info("CPU to MPS conversion tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_mps_device_properties(self):
        """Test MPS device properties."""
        self.info("Testing MPS device properties")
        
        # Get MPS device
        device = torch.device('mps:0')
        
        # Check device type
        expected_type = 'mps' if self.has_mps else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(device.type, expected_type)
        
        # If it's redirected to MPS, check index
        if device.type == 'mps':
            self.assertEqual(device.index, 0)
        
        self.info("MPS device properties tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)


if __name__ == '__main__':
    unittest.main()
