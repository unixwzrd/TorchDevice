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
from TorchDevice.TorchDevice import get_default_device
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
        self.original_default_device = TorchDevice.TorchDevice._default_device
        self.original_cpu_override = TorchDevice.TorchDevice._cpu_override
        
        # Reset the override state
        TorchDevice.TorchDevice._cpu_override = False
        TorchDevice.TorchDevice._default_device = get_default_device()
        
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
        """Test that CPU override works when creating devices."""
        # Get the current default device
        current_default = get_default_device()
        self.info(f"Default device before override: {current_default}")
        
        # Create a device with the special cpu:-1 override
        with TorchDevice.TorchDevice._lock:
            device = torch.device('cpu:-1')
            self.info(f"Created device: {device}")
            
            # Verify that the default device has been set to CPU
            self.assertEqual(TorchDevice.TorchDevice._default_device, 'cpu')
            self.assertTrue(TorchDevice.TorchDevice._cpu_override)
            
            # Verify that the device is a CPU device
            self.assertEqual(device.type, 'cpu')
            
            # Try creating another device with explicit CPU type
            device2 = torch.device('cpu')
            self.info(f"Created another CPU device: {device2}")
            
            # Verify it stays on CPU
            self.assertEqual(device2.type, 'cpu')
            
            # Try an MPS device to make sure it still redirects non-CPU
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device3 = torch.device('mps')
                self.info(f"Created an MPS device: {device3}")
                # With CPU override, non-CPU devices should redirect to CPU
                self.assertEqual(device3.type, 'cpu')
        
        # Check our log output
        log_output = self.log_capture.log_stream.getvalue()
        self.assertIn("CPU override is set", log_output)

    def test_cpu_override_tensor_to(self):
        """Test that CPU override works with tensor.to()."""
        # Create a tensor on the default device
        x = torch.randn(5, 5)
        self.info(f"Created tensor on default device: {x.device}")
        
        # Override to CPU using the special parameter
        with TorchDevice.TorchDevice._lock:
            x = x.to('cpu:-1')
            self.info(f"Moved tensor to device: {x.device}")
            
            # Verify that the default device has been set to CPU
            self.assertEqual(TorchDevice.TorchDevice._default_device, 'cpu')
            self.assertTrue(TorchDevice.TorchDevice._cpu_override)
            
            # Verify the tensor is on CPU
            self.assertEqual(x.device.type, 'cpu')
            
            # Create another tensor and move it to CPU
            y = torch.randn(5, 5)
            y = y.to('cpu')
            self.info(f"Created another tensor and moved to CPU: {y.device}")
            
            # Verify it stays on CPU
            self.assertEqual(y.device.type, 'cpu')
        
        # Check our log output
        log_output = self.log_capture.log_stream.getvalue()
        self.assertIn("CPU override is set", log_output)

    def test_cpu_override_module_to(self):
        """Test that CPU override works with module.to()."""
        # Create a model on the default device
        model = torch.nn.Linear(10, 5)
        self.info(f"Created model with parameters on device: {next(model.parameters()).device}")
        
        # Override to CPU using the special parameter
        with TorchDevice.TorchDevice._lock:
            model = model.to('cpu:-1')
            self.info(f"Moved model to device: {next(model.parameters()).device}")
            
            # Verify that the default device has been set to CPU
            self.assertEqual(TorchDevice.TorchDevice._default_device, 'cpu')
            self.assertTrue(TorchDevice.TorchDevice._cpu_override)
            
            # Verify model parameters are on CPU
            for param in model.parameters():
                self.assertEqual(param.device.type, 'cpu')
            
            # Create another model and explicitly move it to CPU
            model2 = torch.nn.Linear(10, 5)
            model2 = model2.to('cpu')
            self.info(f"Created another model and moved to CPU: {next(model2.parameters()).device}")
            
            # Verify it stays on CPU
            for param in model2.parameters():
                self.assertEqual(param.device.type, 'cpu')
        
        # Check our log output
        log_output = self.log_capture.log_stream.getvalue()
        self.assertIn("CPU override is set", log_output)


if __name__ == '__main__':
    unittest.main() 