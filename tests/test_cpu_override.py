#!/usr/bin/env python
"""
Test file for CPU override functionality in TorchDevice.
Tests the ability to explicitly specify CPU as the default device using 'cpu:-1',
overriding any available accelerators.
"""
import unittest
import torch
from pathlib import Path

# Import TorchDevice to apply patches
import TorchDevice  # Only to trigger patching; do not use directly in test code
from common.test_utils import PrefixedTestCase
from common.log_diff import setup_log_capture, teardown_log_capture

# Suppress linter warnings about unused import - we need to import TorchDevice to apply the patches
_ = TorchDevice


class TestCPUOverride(PrefixedTestCase):
    """
    Test case for the CPU override functionality using 'cpu:-1'.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Call the parent setUp method to set up logging
        super().setUp()
        
        # Set up log capture
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)

    def tearDown(self):
        """Clean up test environment."""
        # Clean up log capture
        teardown_log_capture(self.log_capture)
        
        # Call the parent tearDown method
        super().tearDown()

    def test_cpu_override_device_creation(self):
        """Test that CPU override works when creating devices, and toggling off restores the previous device."""
        # Get the default device before override
        default_before = torch.device('cpu')
        self.info(f"Default device before override: {default_before}")

        # Activate CPU override (toggle ON)
        torch.device('cpu:-1')
        # Create a tensor with explicit CPU override
        t_cpu = torch.randn(2, 2, device='cpu')
        self.assertEqual(t_cpu.device.type, 'cpu')
        # Create a tensor with accelerator device (should be redirected to CPU)
        t_accel = torch.randn(2, 2, device='cuda')
        self.assertEqual(t_accel.device.type, 'cpu')
        # Toggle override OFF
        torch.device('cpu:-1')
        # After override is off, device requests should be redirected to accelerator if available
        t_accel2 = torch.randn(2, 2, device='cuda')
        expected_type = (
            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.assertEqual(t_accel2.device.type, expected_type)

    def test_cpu_override_tensor_to(self):
        """Test that CPU override works with tensor.to(), and toggling off restores the previous device."""
        x = torch.randn(2, 2)
        # Activate CPU override
        torch.device('cpu:-1')
        x_cpu = x.to('cpu')
        self.assertEqual(x_cpu.device.type, 'cpu')
        # Toggle override OFF
        torch.device('cpu:-1')
        # Now .to('cpu') should redirect to accelerator if available
        x_accel = x.to('cpu')
        expected_type = (
            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.assertEqual(x_accel.device.type, expected_type)

    def test_cpu_override_module_to(self):
        """Test that CPU override works with module.to(), and toggling off restores the previous device."""
        model = torch.nn.Linear(10, 5)
        # Activate CPU override
        torch.device('cpu:-1')
        model_cpu = model.to('cpu')
        for param in model_cpu.parameters():
                self.assertEqual(param.device.type, 'cpu')
        # Toggle override OFF
        torch.device('cpu:-1')
        # Now .to('cpu') should redirect to accelerator if available
        model_accel = model.to('cpu')
        expected_type = (
            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        for param in model_accel.parameters():
            self.assertEqual(param.device.type, expected_type)

    def test_cpu_override_toggle_behavior(self):
        """Test toggling CPU override ON and OFF repeatedly, verifying device redirection each time."""
        x = torch.randn(2, 2)
        # Initial: should go to accelerator if available
        x_accel = x.to('cuda')
        expected_type = (
            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.assertEqual(x_accel.device.type, expected_type)

        # Toggle ON
        torch.device('cpu:-1')
        x_cpu = x.to('cuda')
        self.assertEqual(x_cpu.device.type, 'cpu')
        y_cpu = torch.randn(2, 2, device='cuda')
        self.assertEqual(y_cpu.device.type, 'cpu')

            # Toggle OFF
        torch.device('cpu:-1')
        x_accel2 = x.to('cuda')
        self.assertEqual(x_accel2.device.type, expected_type)

        # Toggle ON again
        torch.device('cpu:-1')
        z_cpu = torch.randn(2, 2, device='cuda')
        self.assertEqual(z_cpu.device.type, 'cpu')

        # Toggle OFF again
        torch.device('cpu:-1')
        z_accel = torch.randn(2, 2, device='cuda')
        self.assertEqual(z_accel.device.type, expected_type)


if __name__ == '__main__':
    unittest.main() 