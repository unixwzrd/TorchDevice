#!/usr/bin/env python
"""
Test file for MPS device operations with TorchDevice.
This ensures that all operations work correctly on MPS.
"""
import unittest
import sys
import torch
from pathlib import Path

# Import from common test utilities
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture
from common.test_utils import PrefixedTestCase

# Import TorchDevice to ensure CUDA redirection is set up
import TorchDevice


class TestMPSOperations(PrefixedTestCase):
    """
    Test case for MPS operations to ensure they work correctly with TorchDevice.
    """

    def setUp(self):
        """Set up test environment."""
        super().setUp()

        # Check if MPS is available
        self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

        # Set the device to MPS if available, otherwise CPU
        if self.mps_available:
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.info(f"Using device: {self.device}")

        # Skip tests if MPS is not available
        if not self.mps_available:
            self.skipTest("MPS is not available on this system")

        # Set up log capture
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)

    def tearDown(self):
        teardown_log_capture(self.log_capture)
        super().tearDown()

    def test_mps_tensor_creation(self):
        """Test creating tensors explicitly on MPS using self.device."""
        self.info("Testing explicit MPS tensor creation")

        tensor1 = torch.randn(2, 3, device=self.device)
        tensor2 = torch.zeros(3, 4, device=self.device)
        tensor3 = torch.ones(2, 2, device=self.device)
        tensor4 = torch.tensor([1, 2, 3], device=self.device)

        expected_type = 'mps'
        self.assertEqual(tensor1.device.type, expected_type)
        self.assertEqual(tensor2.device.type, expected_type)
        self.assertEqual(tensor3.device.type, expected_type)
        self.assertEqual(tensor4.device.type, expected_type)

        self.info("Explicit MPS tensor creation tests passed")
        diff_check(self.log_capture)

    def test_mps_tensor_operations(self):
        """Test operations on tensors created explicitly on MPS."""
        self.info("Testing MPS tensor operations")

        a = torch.randn(2, 3, device=self.device)
        b = torch.randn(3, 2, device=self.device)

        c = torch.matmul(a, b)
        d = torch.nn.functional.relu(c)

        expected_type = 'mps'
        self.assertEqual(c.device.type, expected_type)
        self.assertEqual(d.device.type, expected_type)

        self.info("MPS tensor operations tests passed")
        diff_check(self.log_capture)

    def test_mps_nn_operations(self):
        """Test neural network operations on MPS."""
        self.info("Testing MPS neural network operations")

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).to(self.device)

        x = torch.randn(3, 10, device=self.device)
        output = model(x)

        expected_type = 'mps'
        self.assertEqual(output.device.type, expected_type)
        for param in model.parameters():
            self.assertEqual(param.device.type, expected_type)

        self.info("MPS neural network operations tests passed")
        diff_check(self.log_capture)

    def test_cpu_to_mps_conversion(self):
        """
        Test converting tensors from CPU to MPS.

        Two scenarios are covered:
        1. An implicit tensor creation call (without device specified) is now forced to be accelerated,
            so such a tensor should be created on the accelerator.
        2. A tensor created explicitly on CPU using the override ("cpu:-1") should remain on CPU,
            and then can be converted to MPS.
        """
        self.info("Testing CPU-to-MPS conversion")

        # Scenario 1: Implicit creation (no device specified)
        # According to our policy, this will be forced to the accelerator.
        implicit_tensor = torch.randn(2, 3)
        # Expect the tensor to be accelerated.
        self.assertEqual(implicit_tensor.device.type, 'mps')

        # Scenario 2: Forcing a genuine CPU tensor using the override.
        cpu_tensor = torch.randn(2, 3, device="cpu:-1")
        self.assertEqual(cpu_tensor.device.type, 'cpu')
        tmp_dev = torch.device('cpu:-1')

        # Now convert the CPU tensor to MPS.
        converted_tensor = cpu_tensor.to(self.device)
        self.assertEqual(converted_tensor.device.type, 'mps')

        # Convert back to CPU and check values (should redirect to the accelerator device).
        cpu_tensor_again = converted_tensor.cpu()
        self.assertTrue(torch.allclose(converted_tensor, cpu_tensor_again))

        self.info("CPU-to-MPS conversion tests passed")
        diff_check(self.log_capture)

    def test_mps_device_properties(self):
        """Test MPS device properties."""
        self.info("Testing MPS device properties")

        device = torch.device('mps:0')
        expected_type = 'mps'
        self.assertEqual(device.type, expected_type)
        if self.mps_available:
            self.assertEqual(device.index, 0)

        self.info("MPS device properties tests passed")
        diff_check(self.log_capture)


if __name__ == '__main__':
    unittest.main(argv=sys.argv[:1])