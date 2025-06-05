#!/usr/bin/env python
"""
Focused debug test for understanding why device redirection fails
specifically in tensor movement operations.
"""
import unittest
import torch
import sys
import os

# Import the relevant modules
from TorchDevice.core.device import DeviceManager
from TorchDevice.core.tensors import tensor_to_replacement
import TorchDevice

class TestDeviceRedirection(unittest.TestCase):
    """Test device redirection logic in isolation."""
    
    def setUp(self):
        """Set up test environment."""
        # Check if MPS is available
        self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Skip tests if MPS is not available
        if not self.mps_available:
            self.skipTest("MPS is not available on this system")
        
        # Ensure CPU override is False to start
        DeviceManager._cpu_override = False
    
    def test_device_redirection_logic(self):
        """Test the device redirection logic without recursion."""
        # First, verify the direct call works
        print(f"Direct call to DeviceManager.torch_device_replacement('mps'):")
        direct_device = DeviceManager.torch_device_replacement('mps')
        print(f"  Result: {direct_device}")
        
        # Now examine the device redirection logic directly
        print("\nExamining the device redirection logic:")
        device_type = 'mps'
        device_index = None
        
        print(f"  Input device_type: {device_type}")
        print(f"  CPU override active: {DeviceManager._cpu_override}")
        print(f"  Default device type: {DeviceManager._default_device_type}")
        
        # Simulate the core condition from DeviceManager.torch_device_replacement
        should_redirect = not device_type or (device_type != 'cpu' and DeviceManager._cpu_override)
        print(f"  Should redirect: {should_redirect}")
        
        if should_redirect:
            redirected_type = DeviceManager._default_device_type
            print(f"  Redirected to: {redirected_type}")
        else:
            print(f"  No redirection needed, stays as: {device_type}")
        
        # Now let's examine a tensor.to() call and trace what happens
        print("\nTesting tensor.to('mps'):")
        try:
            # Create a tensor
            x = torch.ones(2, 3, device='cpu')
            print(f"  Original tensor device: {x.device}")
            
            # Track CPU override before movement
            print(f"  CPU override before: {DeviceManager._cpu_override()}")
            
            # Now attempt to move it
            # This will trigger tensor_to_replacement -> DeviceManager.torch_device_replacement
            result = x.to('mps')
            print(f"  Result device: {result.device}")
            
            # Track CPU override after movement
            print(f"  CPU override after: {DeviceManager._cpu_override()}")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        # Check if there are any external flags or conditions affecting DeviceManager
        print("\nChecking for external factors:")
        print(f"  CUDA_DEVICE_ORDER env var: {os.environ.get('CUDA_DEVICE_ORDER', 'Not set')}")
        print(f"  CUDA_VISIBLE_DEVICES env var: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"  torch.device.type: {torch.device('mps').type}")
        print(f"  torch.__version__: {torch.__version__}")
        
        # Create a direct PyTorch tensor on MPS to see if PyTorch allows it
        print("\nTrying direct PyTorch tensor creation on MPS:")
        try:
            # Store original method
            original_to = torch.Tensor.to
            
            # Temporarily restore the original PyTorch method
            torch.Tensor.to = DeviceManager.t_Tensor_to
            
            # Try direct creation
            direct_x = torch.ones(2, 3, device='cpu')
            direct_result = direct_x.to('mps')
            print(f"  Direct PyTorch .to() result: {direct_result.device}")
            
            # Restore our patched method
            torch.Tensor.to = tensor_to_replacement
        except Exception as e:
            print(f"  Error with direct call: {e}")
            # Make sure to restore our patched method
            torch.Tensor.to = tensor_to_replacement


if __name__ == '__main__':
    unittest.main()
