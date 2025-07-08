#!/usr/bin/env python
import logging
import unittest
import sys
from pathlib import Path
import torch
import TorchDevice
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture
from common.testing_utils import PrefixedTestCase, set_deterministic_seed, devices_equivalent

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Ensure this configuration is applied
)
logger = logging.getLogger(__name__)

# Define a fixed seed for reproducible tests
SEED = 42

class TestCPUOperations(PrefixedTestCase):
    """
    Test case for CPU operations to ensure they work correctly with TorchDevice.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Call the parent setUp method to set up logging
        super().setUp()
        
        # Explicitly set seeds for deterministic behavior
        set_deterministic_seed(SEED)
        
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
        # Ensure CPU override is OFF to start, so we have a known state.
        # We do this by checking the default device; if it's CPU, we toggle the override.
        default_device_type = torch.get_default_device().type
        if default_device_type == 'cpu':
            self.info("CPU override was ON, toggling OFF first.")
            torch.device('cpu:-1') # Toggle OFF

        self.info("Toggling CPU override ON for test.")
        # Toggle override ON
        torch.device('cpu:-1')

        # Verify that all device requests now return 'cpu' type.
        self.assertEqual(torch.device('cpu').type, 'cpu')
        self.assertEqual(torch.device('mps').type, 'cpu')
        self.assertEqual(torch.device('cuda').type, 'cpu')
        self.info("All devices correctly forced to 'cpu' type.")

        # Toggle override OFF to clean up for subsequent tests.
        self.info("Toggling CPU override OFF.")
        torch.device('cpu:-1')

    def test_explicit_accelerator_requests(self):
        """Test that explicit accelerator requests are honored when override is OFF."""
        # Ensure CPU override is OFF. We toggle it twice to be sure of the final state.
        torch.device('cpu:-1')
        torch.device('cpu:-1')

        # Determine the expected default accelerator type.
        default_accelerator_type = torch.get_default_device().type

        self.info(f"Default accelerator type for this system is '{default_accelerator_type}'")

        # When override is OFF, all device requests should be redirected to the default accelerator.
        self.info("Verifying that 'cpu', 'mps', and 'cuda' requests all redirect to the default accelerator.")
        self.assertEqual(torch.device('cpu').type, default_accelerator_type)
        self.assertEqual(torch.device('mps').type, default_accelerator_type)
        self.assertEqual(torch.device('cuda').type, default_accelerator_type)
        # Toggle override OFF
        torch.device('cpu:-1')


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

        # Determine the default accelerator for this system.
        self.expected_type = torch.get_default_device().type

        if self.expected_type == 'cpu':
            self.skipTest("This test requires an accelerator (MPS or CUDA) to run.")

        # We will request 'mps' and expect TorchDevice to redirect it to the actual default accelerator.
        self.device = torch.device('mps')
        self.logger.info("Requesting 'mps', expecting to get '%s'", self.expected_type)

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
        expected_type = torch.get_default_device()
        if not isinstance(expected_type, str):
            expected_type = expected_type.type
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
    unittest.main(argv=sys.argv[:1])
