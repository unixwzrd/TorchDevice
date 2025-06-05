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
from common.test_utils import PrefixedTestCase, set_deterministic_seed
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
        # Store original values to restore later
        self._original_cpu_override = DeviceManager._cpu_override
        # Reset CPU override
        DeviceManager._cpu_override = False
        
        # Set deterministic seed for reproducible tests
        set_deterministic_seed(42)
        
        # Set up log capture
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)
        
        self.info("Default device: %s" % DeviceManager.get_default_device())
        self.info("CPU Override: %s" % DeviceManager._cpu_override)

    def tearDown(self):
        """Restore original state."""
        teardown_log_capture(self.log_capture)
        DeviceManager._cpu_override = self._original_cpu_override
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
        default_device = DeviceManager.get_default_device()
        self.info("Creating tensors with explicit device: %s" % default_device)
        
        t4 = torch.tensor([1, 2, 3], device=default_device)
        t5 = torch.zeros(3, device=default_device)
        t6 = torch.ones(3, device=default_device)
        
        self.info("t4 device: %s" % t4.device)
        self.info("t5 device: %s" % t5.device)
        self.info("t6 device: %s" % t6.device)
        
        self.assertEqual(t4.device, default_device)
        self.assertEqual(t5.device, default_device)
        self.assertEqual(t6.device, default_device)

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
            
            # Should be on MPS if available and not CPU overridden
            if not DeviceManager._cpu_override:
                self.assertEqual(t_mps.device.type, 'mps')
        
        if has_cuda:
            # Test CUDA to default device redirection
            self.info("Testing CUDA tensor creation")
            t_cuda = torch.tensor([1, 2, 3], device='cuda')
            self.info("CUDA tensor device: %s" % t_cuda.device)
            
            # Should be on CUDA if available and not CPU overridden
            if not DeviceManager._cpu_override:
                self.assertEqual(t_cuda.device.type, 'cuda')
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_cpu_override(self):
        """Test that CPU override works correctly."""
        # Enable CPU override
        DeviceManager._cpu_override = True
        self.info("CPU Override enabled: %s" % DeviceManager._cpu_override)
        
        # Create tensors with different devices
        self.info("Creating tensors with CPU override enabled")
        
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
        
        # All should be on the same default device
        self.assertEqual(t1.device, t2.device)
        self.assertEqual(t2.device, t3.device)
        
        # Test with explicit device
        default_device = DeviceManager.get_default_device()
        self.info("Creating random tensors with explicit device: %s" % default_device)
        
        t4 = torch.rand(3, device=default_device)
        t5 = torch.randn(3, device=default_device)
        t6 = torch.randint(0, 10, (3,), device=default_device)
        
        self.info("t4 (rand) device: %s" % t4.device)
        self.info("t5 (randn) device: %s" % t5.device)
        self.info("t6 (randint) device: %s" % t6.device)
        
        self.assertEqual(t4.device, default_device)
        self.assertEqual(t5.device, default_device)
        self.assertEqual(t6.device, default_device)


if __name__ == '__main__':
    unittest.main()
