#!/usr/bin/env python3
"""
Test script for unified tensor creation wrapper functionality.

This test verifies that tensor creation works consistently across
different PyTorch functions that have been patched with the 
unified tensor_creation_wrapper.
"""

import os
import sys
import unittest
import torch
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

# Import TorchDevice - this should automatically patch torch
import TorchDevice  # noqa
from TorchDevice.core.device import DeviceManager
from common.testing_utils import PrefixedTestCase, set_deterministic_seed
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestUnifiedTensorCreation(PrefixedTestCase):
    """Test the unified tensor creation wrapper functionality."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Store original CPU override state
        self._initial_cpu_override_state = DeviceManager.cpu_override()
        # Ensure CPU override is OFF at the start of each test method by default, if it was ON.
        if self._initial_cpu_override_state:
            torch.device("cpu:-1") # Toggle it OFF if it was ON
        
        # Set deterministic seed for reproducible tests
        set_deterministic_seed(42)
        
        # Set up log capture
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)
        
        self.info("Default device: %s" % torch.get_default_device())
        self.info("Current default device type (CPU override intended to be False at test start): %s" % torch.get_default_device().type)

    def tearDown(self):
        """Restore original state."""
        teardown_log_capture(self.log_capture)
        # Restore original CPU override state if it was changed by the test
        if DeviceManager.cpu_override() != self._initial_cpu_override_state:
            torch.device("cpu:-1") # Toggle to restore
        super().tearDown()

    def test_basic_tensor_creation(self):
        """Test that basic tensor creation functions are patched consistently."""
        # Create tensors with different creation functions
        self.info("Creating tensors with different creation functions")
        
        # Test with default device (no device specified)
        t1 = torch.tensor([1, 2, 3])
        t2 = torch.zeros(3)
        t3 = torch.ones(3)
        
        # All should be on the same default device
        self.info("t1 device: %s" % t1.device)
        self.info("t2 device: %s" % t2.device)
        self.info("t3 device: %s" % t3.device)
        
        self.assertEqual(t1.device, t2.device)
        self.assertEqual(t2.device, t3.device)
        
        # Test with explicit device
        default_device = torch.get_default_device()
        self.info("Creating tensors with explicit device: %s" % default_device)
        
        t4 = torch.tensor([1, 2, 3], device=default_device)
        t5 = torch.zeros(3, device=default_device)
        t6 = torch.ones(3, device=default_device)
        
        self.info("t4 device: %s" % t4.device)
        self.info("t5 device: %s" % t5.device)
        self.info("t6 device: %s" % t6.device)
        
        self.assertEqual(t4.device.type, default_device.type, f"t4 device type {t4.device.type} vs default {default_device.type}")
        self.assertTrue(t4.device.index == default_device.index or (default_device.index is None and t4.device.index == 0) or (t4.device.index is None and default_device.index == 0), f"t4 device {t4.device} vs default {default_device}")
        self.assertEqual(t5.device.type, default_device.type, f"t5 device type {t5.device.type} vs default {default_device.type}")
        self.assertTrue(t5.device.index == default_device.index or (default_device.index is None and t5.device.index == 0) or (t5.device.index is None and default_device.index == 0), f"t5 device {t5.device} vs default {default_device}")
        self.assertEqual(t6.device.type, default_device.type, f"t6 device type {t6.device.type} vs default {default_device.type}")
        self.assertTrue(t6.device.index == default_device.index or (default_device.index is None and t6.device.index == 0) or (t6.device.index is None and default_device.index == 0), f"t6 device {t6.device} vs default {default_device}")

    def test_device_redirection(self):
        """Test that device redirection works correctly."""
        # Get available devices
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch, 'mps') and torch.backends.mps.is_available()
        
        self.info("CUDA available: %s" % has_cuda)
        self.info("MPS available: %s" % has_mps)
        
        if has_mps:
            # Test MPS to default device redirection
            self.info("Testing MPS tensor creation")
            t_mps = torch.tensor([1, 2, 3], device='mps')
            self.info("MPS tensor device: %s" % t_mps.device)
            
            # Should be on MPS if available and CPU override is OFF (ensured by setUp)
            self.assertEqual(t_mps.device.type, 'mps')
        
        if has_cuda:
            # Test CUDA to default device redirection
            self.info("Testing CUDA tensor creation")
            t_cuda = torch.tensor([1, 2, 3], device='cuda')
            self.info("CUDA tensor device: %s" % t_cuda.device)
            
            # On an MPS machine, TorchDevice redirects 'cuda' device requests to 'mps'.
            # So, the resulting tensor's device type should be 'mps'.
            # CPU override is OFF (ensured by setUp).
            self.assertEqual(t_cuda.device.type, 'mps', "Tensor requested on 'cuda' should be on 'mps' due to TorchDevice redirection on an MPS system.")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_cpu_override(self):
        """Test that CPU override works correctly."""
        # setUp ensures CPU override is OFF. Now, toggle it ON for this test.
        torch.device("cpu:-1") 
        
        # Verify that CPU is now the default device type as an effect of the override.
        current_default_device_type = torch.get_default_device().type
        self.assertEqual(current_default_device_type, 'cpu', 
                         f"After toggling CPU override ON, default device type should be 'cpu', but got '{current_default_device_type}'")
        self.info("CPU Override toggled ON for test. Current default device type: %s" % current_default_device_type)
        
        # Create tensors with different devices
        self.info("Creating tensors with CPU override enabled (behavioral check)")
        
        t1 = torch.tensor([1, 2, 3])  # default
        t2 = torch.zeros(3, device='cpu')  # explicit CPU
        
        # These should be redirected to CPU
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch, 'mps') and torch.backends.mps.is_available()
        
        if has_mps:
            t_mps = torch.tensor([1, 2, 3], device='mps')
            self.info("MPS tensor with CPU override device: %s" % t_mps.device)
            self.assertEqual(t_mps.device.type, 'cpu')
        
        if has_cuda:
            t_cuda = torch.tensor([1, 2, 3], device='cuda')
            self.info("CUDA tensor with CPU override device: %s" % t_cuda.device)
            self.assertEqual(t_cuda.device.type, 'cpu')
        
        # All should be on CPU
        self.info("t1 device: %s" % t1.device)
        self.info("t2 device: %s" % t2.device)
        
        self.assertEqual(t1.device.type, 'cpu')
        self.assertEqual(t2.device.type, 'cpu')
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_random_tensor_creation(self):
        """Test that random tensor creation functions are patched consistently."""
        self.info("Testing random tensor creation functions")
        
        # Test random tensor creation
        t1 = torch.rand(3)
        t2 = torch.randn(3)
        t3 = torch.randint(0, 10, (3,))
        
        self.info("Random tensor devices:")
        self.info("t1 (rand) device: %s" % t1.device)
        self.info("t2 (randn) device: %s" % t2.device)
        self.info("t3 (randint) device: %s" % t3.device)
        
        self.assertEqual(t1.device.type, t2.device.type, f"t1 device type {t1.device.type} vs t2 {t2.device.type}")
        self.assertTrue(t1.device.index == t2.device.index or (t2.device.index is None and t1.device.index == 0) or (t1.device.index is None and t2.device.index == 0), f"t1 device {t1.device} vs t2 {t2.device}")
        self.assertEqual(t2.device.type, t3.device.type, f"t2 device type {t2.device.type} vs t3 {t3.device.type}")
        self.assertTrue(t2.device.index == t3.device.index or (t3.device.index is None and t2.device.index == 0) or (t2.device.index is None and t3.device.index == 0), f"t2 device {t2.device} vs t3 {t3.device}")
        
        # Test with explicit device
        default_device = torch.get_default_device()
        self.info("Creating random tensors with explicit device: %s" % default_device)
        
        t4 = torch.rand(3, device=default_device)
        t5 = torch.randn(3, device=default_device)
        t6 = torch.randint(0, 10, (3,), device=default_device)
        
        self.info("t4 (rand) device: %s" % t4.device)
        self.info("t5 (randn) device: %s" % t5.device)
        self.info("t6 (randint) device: %s" % t6.device)
        
        self.assertEqual(t4.device.type, default_device.type)
        self.assertTrue(t4.device.index == default_device.index or (default_device.index is None and t4.device.index == 0) or (t4.device.index is None and default_device.index == 0), f"t4 device {t4.device} vs default {default_device}")
        self.assertEqual(t5.device.type, default_device.type)
        self.assertTrue(t5.device.index == default_device.index or (default_device.index is None and t5.device.index == 0) or (t5.device.index is None and default_device.index == 0), f"t5 device {t5.device} vs default {default_device}")
        self.assertEqual(t6.device.type, default_device.type)
        self.assertTrue(t6.device.index == default_device.index or (default_device.index is None and t6.device.index == 0) or (t6.device.index is None and default_device.index == 0), f"t6 device {t6.device} vs default {default_device}")


if __name__ == '__main__':
    unittest.main(argv=sys.argv[:1])
