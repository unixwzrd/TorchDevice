#!/usr/bin/env python
"""
Debug test for device manager issues.
This isolates the DeviceManager.torch_device_replacement function for debugging.
"""
import unittest
import torch
import os
import sys

# Add debugging output for the CPU override logic
from TorchDevice.core.device import DeviceManager
from TorchDevice.core.logger import log_info

class TestDeviceManager(unittest.TestCase):
    """Test the DeviceManager's device replacement logic."""

    def setUp(self):
        """Set up test environment."""
        # Check if MPS is available
        self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Skip tests if MPS is not available
        if not self.mps_available:
            self.skipTest("MPS is not available on this system")
            
    def test_device_override_state(self):
        """Test the state of the CPU override flag and device redirection logic."""
        # Check initial state
        print(f"Initial CPU override state: {DeviceManager.cpu_override()}")
        print(f"Default device type: {DeviceManager._default_device_type}")
        
        # Test device redirection with MPS
        device_mps = DeviceManager.torch_device_replacement('mps')
        print(f"Requested 'mps', got: {device_mps}")
        
        # Test device redirection with CUDA
        device_cuda = DeviceManager.torch_device_replacement('cuda')
        print(f"Requested 'cuda', got: {device_cuda}")
        
        # Test device redirection with CPU
        device_cpu = DeviceManager.torch_device_replacement('cpu')
        print(f"Requested 'cpu', got: {device_cpu}")
        
        # Check if override is active
        print(f"Current CPU override state: {DeviceManager.cpu_override()}")
        
        # Try resetting the override if it's on
        if DeviceManager.cpu_override():
            print("Attempting to reset CPU override")
            # Force reset the override flag
            DeviceManager._cpu_override = False
            DeviceManager._default_device_type = DeviceManager._detect_default_device_type()
            print(f"After reset - CPU override: {DeviceManager.cpu_override()}")
            print(f"After reset - Default device: {DeviceManager._default_device_type}")
            
            # Test again
            device_mps_after = DeviceManager.torch_device_replacement('mps')
            print(f"After reset - Requested 'mps', got: {device_mps_after}")

if __name__ == '__main__':
    unittest.main()
