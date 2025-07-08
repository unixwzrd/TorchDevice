#!/usr/bin/env python
"""
Test file for CPU override functionality in TorchDevice.
Tests the ability to explicitly specify CPU as the default device using 'cpu:-1',
overriding any available accelerators.
"""
import unittest
import sys
import torch
from pathlib import Path

from common.test_utils import PrefixedTestCase
from common.log_diff import setup_log_capture, teardown_log_capture


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
        # Ensure override is OFF to start
        if torch.get_default_device().type == 'cpu':
            torch.device('cpu:-1')  # Toggle OFF
        
        expected_device = torch.get_default_device().type
        self.assertNotEqual(expected_device, 'cpu')

        # Activate CPU override (toggle ON)
        torch.device('cpu:-1')
        self.assertEqual(torch.device('cuda').type, 'cpu')
        self.assertEqual(torch.device('cpu').type, 'cpu')

        # Toggle override OFF
        torch.device('cpu:-1')
        self.assertEqual(torch.device('cuda').type, expected_device)

    def test_cpu_override_tensor_to(self):
        """Test that CPU override works with tensor.to(), and toggling off restores the previous device."""
        # Ensure override is OFF to start
        if torch.get_default_device().type == 'cpu':
            torch.device('cpu:-1')  # Toggle OFF

        expected_device = torch.get_default_device().type
        x = torch.randn(2, 2)

        # Activate CPU override
        torch.device('cpu:-1')
        x_cpu = x.to('cuda')
        self.assertEqual(x_cpu.device.type, 'cpu')

        # Toggle override OFF
        torch.device('cpu:-1')
        x_accel = x.to('cuda')
        self.assertEqual(x_accel.device.type, expected_device)

    def test_cpu_override_module_to(self):
        """Test that CPU override works with module.to(), and toggling off restores the previous device."""
        # Ensure override is OFF to start
        if torch.get_default_device().type == 'cpu':
            torch.device('cpu:-1')  # Toggle OFF

        expected_device = torch.get_default_device().type
        model = torch.nn.Linear(10, 5)

        # Activate CPU override
        torch.device('cpu:-1')
        model_cpu = model.to('cuda')
        for param in model_cpu.parameters():
            self.assertEqual(param.device.type, 'cpu')

        # Toggle override OFF
        torch.device('cpu:-1')
        model_accel = model.to('cuda')
        for param in model_accel.parameters():
            self.assertEqual(param.device.type, expected_device)

    def test_cpu_override_toggle_behavior(self):
        """Test toggling CPU override ON and OFF repeatedly, verifying device redirection each time."""
        # Ensure override is OFF to start
        if torch.get_default_device().type == 'cpu':
            torch.device('cpu:-1')  # Toggle OFF

        expected_device = torch.get_default_device().type
        self.assertNotEqual(expected_device, 'cpu')

        # Toggle ON
        torch.device('cpu:-1')
        self.assertEqual(torch.device('cuda').type, 'cpu')

        # Toggle OFF
        torch.device('cpu:-1')
        self.assertEqual(torch.device('cuda').type, expected_device)

        # Toggle ON again
        torch.device('cpu:-1')
        self.assertEqual(torch.device('cuda').type, 'cpu')

        # Toggle OFF again
        torch.device('cpu:-1')
        self.assertEqual(torch.device('cuda').type, expected_device)


if __name__ == '__main__':
    unittest.main(argv=sys.argv[:1])