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
import TorchDevice
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
        
        # Record the default device before our tests
        self.original_default_device = TorchDevice.TorchDevice.get_default_device()
        self.original_cpu_override = TorchDevice.TorchDevice.cpu_override_set()
        
        # Reset the override state
        TorchDevice.TorchDevice._cpu_override = False
        TorchDevice.TorchDevice._default_device = TorchDevice.TorchDevice.get_default_device()
        
        # Set up log capture
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)

    def tearDown(self):
        """Clean up test environment."""
        # Restore the original state
        with TorchDevice.TorchDevice._lock:
            TorchDevice.TorchDevice._default_device = self.original_default_device
            TorchDevice.TorchDevice._cpu_override = self.original_cpu_override
        
        # Clean up log capture
        teardown_log_capture(self.log_capture)
        
        # Call the parent tearDown method
        super().tearDown()

    def test_cpu_override_device_creation(self):
        """Test that CPU override works when creating devices, and toggling off restores the previous device."""
        # Get the current default device
        host_default = TorchDevice.TorchDevice.get_default_device()
        self.info(f"Default device before override: {host_default}")

        # Create a device with the special cpu:-1 override (toggle ON)
        with TorchDevice.TorchDevice._lock:
            device = torch.device('cpu:-1')
            default_device = TorchDevice.TorchDevice.get_default_device()
            self.info(f"Created device: {device}")

            # Verify that the default device has been set to CPU
            self.assertEqual(default_device, 'cpu')
            self.assertNotEqual(host_default, default_device)
            self.assertEqual(device.type, 'cpu')

            # Try creating another device with explicit CPU type
            device2 = torch.device('cpu')
            self.info(f"Created another CPU device: {device2}")
            self.assertEqual(device2.type, 'cpu')

            # Try an acclerated device to make sure it still redirects non-CPU
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.backends.cuda.is_available():
                device = 'cuda'
            device3 = torch.device(device)
            self.info(f"Created an {device} device: {device3}")
            self.assertEqual(device3.type, 'cpu')
            # Toggle OFF
            device_off = torch.device('cpu:-1')
            self.info(f"Toggled override OFF, device: {device_off}")
            current_default = TorchDevice.TorchDevice.get_default_device()
            self.assertFalse(TorchDevice.TorchDevice.cpu_override_set())
            self.assertEqual(current_default, host_default)


    def test_cpu_override_tensor_to(self):
        """Test that CPU override works with tensor.to(), and toggling off restores the previous device."""
        x = torch.randn(5, 5)
        self.info(f"Created tensor on default device: {x.device}")
        with TorchDevice.TorchDevice._lock:
            x = x.to('cpu:-1')
            self.info(f"Moved tensor to device: {x.device}")
            self.assertEqual(TorchDevice.TorchDevice.get_default_device(), 'cpu')
            self.assertTrue(TorchDevice.TorchDevice.cpu_override_set())
            self.assertEqual(x.device.type, 'cpu')

            y = torch.randn(5, 5)
            y = y.to('cpu')
            self.info(f"Created another tensor and moved to CPU: {y.device}")
            self.assertEqual(y.device.type, 'cpu')
            # Toggle OFF
            x = x.to('cpu:-1')
            self.info(f"Toggled override OFF, tensor device: {x.device}")
            self.assertFalse(TorchDevice.TorchDevice.cpu_override_set())


    def test_cpu_override_module_to(self):
        """Test that CPU override works with module.to(), and toggling off restores the previous device."""
        model = torch.nn.Linear(10, 5)
        self.info(f"Created model with parameters on device: {next(model.parameters()).device}")
        with TorchDevice.TorchDevice._lock:
            model = model.to('cpu:-1')
            self.info(f"Moved model to device: {next(model.parameters()).device}")
            self.assertEqual(TorchDevice.TorchDevice.get_default_device(), 'cpu')
            self.assertTrue(TorchDevice.TorchDevice.cpu_override_set())
            for param in model.parameters():
                self.assertEqual(param.device.type, 'cpu')
            # New model should have parameters on CPU
            model2 = torch.nn.Linear(10, 5)
            model2 = model2.to('cpu')
            self.info(f"Created another model and moved to CPU: {next(model2.parameters()).device}")
            for param in model2.parameters():
                self.assertEqual(param.device.type, 'cpu')
            # Toggle OFF
            model = model.to('cpu:-1')
            self.info(f"Toggled override OFF, model device: {next(model.parameters()).device}")
            self.assertFalse(TorchDevice.TorchDevice.cpu_override_set())



if __name__ == '__main__':
    unittest.main() 