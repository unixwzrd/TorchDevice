#!/usr/bin/env python
import logging
import unittest
from pathlib import Path

import TorchDevice

import torch
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture
from common.test_utils import PrefixedTestCase

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
        # Remove device override from setUp
        # self.device = torch.device('cpu:-1')
        # self.info("Using device: %s", self.device)

    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        teardown_log_capture(self.log_capture)
        print("\n" + "=" * 80)
        print(f"Finished test: {self._testMethodName}")
        print("=" * 80)

    def test_cpu_tensor_creation(self):
        """Test creating tensors explicitly on CPU."""
        # Toggle CPU override ON for this test
        self.device = torch.device('cpu:-1')
        self.info("Testing CPU tensor creation")
        tensor1 = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        tensor2 = torch.ones((2, 3), device=self.device)
        self.assertEqual(tensor1.device.type, 'cpu')
        self.assertEqual(tensor2.device.type, 'cpu')
        self.info("CPU tensor creation tests passed")
        self.log_capture.log_stream.getvalue()
        diff_check(self.log_capture)
        # Toggle CPU override OFF after test
        torch.device('cpu:-1')
    
    def test_cpu_tensor_operations(self):
        """Test operations on CPU tensors."""
        # Toggle CPU override ON for this test
        self.device = torch.device('cpu:-1')
        self.info("Testing CPU tensor operations")
        a = torch.randn(10, device=self.device)
        b = torch.randn(10, device=self.device)
        c = a + b
        d = a * b
        e = torch.matmul(a, b)
        self.assertEqual(c.device.type, 'cpu')
        self.assertEqual(d.device.type, 'cpu')
        self.assertEqual(e.device.type, 'cpu')
        self.info("CPU tensor operations tests passed")
        self.log_capture.log_stream.getvalue()
        diff_check(self.log_capture)
        # Toggle CPU override OFF after test
        torch.device('cpu:-1')
    
    def test_cpu_nn_operations(self):
        """Test neural network operations on CPU."""
        # Toggle CPU override ON for this test
        self.device = torch.device('cpu:-1')
        self.info("Testing CPU neural network operations")
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).to(self.device)
        x = torch.randn(3, 10, device=self.device)
        output = model(x)
        self.assertEqual(output.device.type, 'cpu')
        for param in model.parameters():
            self.assertEqual(param.device.type, 'cpu')
        self.info("CPU neural network operations tests passed")
        self.log_capture.log_stream.getvalue()
        diff_check(self.log_capture)
        # Toggle CPU override OFF after test
        torch.device('cpu:-1')

    def test_override_forces_all_devices_to_cpu(self):
        """Test that CPU override forces all device requests to CPU."""
        # Toggle override ON
        torch.device('cpu:-1')
        # Try to create tensors on all device types
        t_cpu = torch.tensor([1], device='cpu')
        t_mps = torch.tensor([1], device='mps')
        t_cuda = torch.tensor([1], device='cuda')
        # All should be on CPU
        self.assertEqual(t_cpu.device.type, 'cpu')
        self.assertEqual(t_mps.device.type, 'cpu')
        self.assertEqual(t_cuda.device.type, 'cpu')
        # Toggle override OFF
        torch.device('cpu:-1')

    def test_explicit_accelerator_requests(self):
        """Test that explicit accelerator requests are honored when override is OFF."""
        # Ensure override is OFF
        torch.device('cpu:-1')  # ON
        torch.device('cpu:-1')  # OFF
        # Try to create tensors on all device types
        t_cpu = torch.tensor([1], device='cpu')
        t_mps = torch.tensor([1], device='mps')
        t_cuda = torch.tensor([1], device='cuda')
        # Print device types for manual inspection
        print("t_cpu.device:", t_cpu.device)
        print("t_mps.device:", t_mps.device)
        print("t_cuda.device:", t_cuda.device)
        # Optionally, assert based on your system's default accelerator
        # For example, if MPS is available:
        # self.assertEqual(t_mps.device.type, 'mps')
        # self.assertEqual(t_cuda.device.type, 'mps')


class TestMPSOperations(PrefixedTestCase):
    """
    Test case for MPS operations to ensure they work correctly with TorchDevice.
    """
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        
        print("\n" + "=" * 80)
        print(f"Starting test: {self._testMethodName}")
        print("=" * 80)
        
        # Set up log capture with a unique test name.
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)   
        self.logger = logging.getLogger(__name__)

        # Determine the available hardware - use MPS if available
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        self.has_cuda = torch.cuda.is_available()
        
        # Skip tests if neither MPS nor CUDA is available
        if not self.has_mps and not self.has_cuda:
            self.skipTest("Neither MPS nor CUDA is available on this machine")
            
        # Create device - this will be redirected to the appropriate type by TorchDevice
        self.device = torch.device('mps')
        self.expected_type = 'mps' if self.has_mps else 'cuda' if self.has_cuda else 'cpu'
        self.logger.info("Using device: %s (expected type: %s)", self.device, self.expected_type)

    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        teardown_log_capture(self.log_capture)
        
        # Print a footer for the test
        print("\n" + "=" * 80)
        print(f"Finished test: {self._testMethodName}")
        print("=" * 80)

    def test_cpu_to_mps_conversion(self):
        """Test converting tensors from CPU to MPS."""
        self.logger.info("Using device: %s", self.device)
        self.logger.info("Testing CPU to MPS conversion")

        # Toggle CPU override ON
        torch.device('cpu:-1')

        # Create a tensor on CPU
        cpu_tensor = torch.randn(10, device='cpu:0')
        self.assertEqual(cpu_tensor.device.type, 'cpu')

        # Toggle CPU override OFF (allow moving to MPS)
        torch.device('cpu:-1')

        # Convert to MPS
        mps_tensor = cpu_tensor.to('mps')
        expected_type = 'mps' if self.has_mps else 'cuda' if self.has_cuda else 'cpu'
        self.assertEqual(mps_tensor.device.type, expected_type)

        # Verify data is preserved
        cpu_data = cpu_tensor.tolist()
        mps_data = mps_tensor.cpu().tolist()
        self.assertEqual(cpu_data, mps_data)

    def test_mps_device_properties(self):
        """Test MPS device properties."""
        self.logger.info("Testing MPS device properties")
        
        # Get MPS device
        device = torch.device('mps')  # Use explicit MPS device
        
        # Check device type
        self.assertEqual(device.type, self.expected_type)
        
        # If it's redirected to MPS, check index is either None or 0
        if device.type == 'mps':
            self.assertIn(device.index, [None, 0])
        
        self.logger.info("MPS device properties tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_mps_tensor_creation(self):
        """Test creating tensors explicitly on MPS."""
        self.logger.info("Testing MPS tensor creation")
        
        # Create tensors with different methods
        tensor1 = torch.randn(10, device=self.device)  # Use self.device
        tensor2 = torch.zeros(10, device=self.device)
        tensor3 = torch.ones(10, device=self.device)
        tensor4 = torch.tensor([1, 2, 3], device=self.device)
        
        # Verify they're on the expected device
        self.assertEqual(tensor1.device.type, self.expected_type)
        self.assertEqual(tensor2.device.type, self.expected_type)
        self.assertEqual(tensor3.device.type, self.expected_type)
        self.assertEqual(tensor4.device.type, self.expected_type)
        
        self.logger.info("MPS tensor creation tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_mps_tensor_operations(self):
        """Test operations on MPS tensors."""
        self.logger.info("Testing MPS tensor operations")
        
        # Create tensors
        a = torch.randn(10, device=self.device)  # Use self.device
        b = torch.randn(10, device=self.device)
        
        # Test basic operations
        c = a + b
        d = a * b
        e = torch.matmul(a, b)
        
        # Verify results are on the expected device
        self.assertEqual(c.device.type, self.expected_type)
        self.assertEqual(d.device.type, self.expected_type)
        self.assertEqual(e.device.type, self.expected_type)
        
        self.logger.info("MPS tensor operations tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_mps_nn_operations(self):
        """Test neural network operations on MPS."""
        self.logger.info("Testing MPS neural network operations")
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).to(self.device)  # Use self.device
        
        # Create input
        x = torch.randn(3, 10, device=self.device)  # Use self.device
        
        # Forward pass
        output = model(x)
        
        # Verify output is on the expected device
        self.assertEqual(output.device.type, self.expected_type)
        
        # Check model parameters are on the expected device
        for param in model.parameters():
            self.assertEqual(param.device.type, self.expected_type)
        
        self.logger.info("MPS neural network operations tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)


if __name__ == '__main__':
    unittest.main()
