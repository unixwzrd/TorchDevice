"""
Tests for torch.device type compatibility.

This module contains tests that verify TorchDevice correctly maintains 
type compatibility with the original torch.device class.
"""
import unittest
import torch
# Import TorchDevice to apply patches
import TorchDevice  # noqa: F401
from common.test_utils import PrefixedTestCase, set_deterministic_seed


# Define a fixed seed for reproducible tests
SEED = 42


class DeviceTypeCompatibilityTest(PrefixedTestCase):
    """Tests for device type compatibility after patching."""

    def setUp(self):
        # Call parent setUp to enable logging
        super().setUp()
        
        # Explicitly set seeds for deterministic behavior
        set_deterministic_seed(SEED)
        
        # Create device instances
        self.cpu_device = torch.device('cpu')
        self.default_device = torch.device(
            'cuda' if torch.cuda.is_available() 
            else 'mps' if torch.backends.mps.is_available() 
            else 'cpu'
        )

    def test_device_type_check(self):
        """Test that isinstance still works correctly with torch.device."""
        # Get the original device type
        original_device_type = torch.device('cpu').__class__
        
        # Verify device instances are recognized as torch.device
        self.assertIsInstance(self.cpu_device, original_device_type)
        self.assertIsInstance(self.default_device, original_device_type)
        
        # Test with dynamically created device
        new_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.assertIsInstance(new_device, original_device_type)
        
        # Log info for clarity
        self.info(f"Original device type: {original_device_type}")
        self.info(f"Patched device function type: {type(torch.device)}")
        self.info(f"CPU device type: {type(self.cpu_device)}")
        self.info(f"Default device type: {type(self.default_device)}")
    
    def test_mock_library_isinstance_check(self):
        """Test a mock library function that uses isinstance with torch.device."""
        def external_library_function(param):
            """Simulates an external library function that checks isinstance(param, torch.device)."""
            if isinstance(param, type(torch.device('cpu'))):
                return True
            return False
        # Test with device objects
        self.assertTrue(external_library_function(self.cpu_device))
        self.assertTrue(external_library_function(self.default_device))
        # Test with non-device objects
        self.assertFalse(external_library_function("not a device"))
        self.assertFalse(external_library_function(42))
    
    def external_library_function(self, param):
        return isinstance(param, type(torch.device('cpu')))

    def test_library_isinstance_check(self):
        self.assertTrue(self.external_library_function(self.cpu_device))

    def param_default_check(self):
        # Define a mock parameter class with a default attribute
        class ParamDefault:
            def __init__(self, default):
                self.default = default
        param_default = ParamDefault(torch.device('cpu'))
        return isinstance(param_default.default, type(torch.device('cpu')))

    def test_with_default_param_check(self):
        self.assertTrue(self.param_default_check())


if __name__ == '__main__':
    # Enable to run just this test
    unittest.main() 