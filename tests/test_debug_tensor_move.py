#!/usr/bin/env python
"""
Debug test for tensor movement and DeviceManager interactions.
This isolates the exact issue with tensor movement functions.
"""
import unittest
import torch
import os
import sys

# Add debugging output
from TorchDevice.core.device import DeviceManager
from TorchDevice.core.tensors import tensor_to_replacement
from TorchDevice.core.logger import log_info, auto_log

class TestTensorMovement(unittest.TestCase):
    """Test the interaction between tensor movement and DeviceManager."""

    def setUp(self):
        """Set up test environment."""
        # Check if MPS is available
        self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Skip tests if MPS is not available
        if not self.mps_available:
            self.skipTest("MPS is not available on this system")
    
    def test_device_state_in_tensor_movement(self):
        """Test the DeviceManager state during tensor movement."""
        # Create a tensor on CPU
        cpu_tensor = torch.ones(2, 3, device="cpu")
        
        # Check CPU override state before any operation
        print(f"Initial CPU override state: {DeviceManager._cpu_override}")
        print(f"Default device type: {DeviceManager._default_device_type}")
        
        # Direct device redirection test
        print("\nDirect device redirection test:")
        direct_device = DeviceManager.torch_device_replacement('mps')
        print(f"Direct call - Requested 'mps', got: {direct_device}")
        
        # Tensor movement test via tensor_to_replacement function
        print("\nTensor movement via tensor_to_replacement:")
        # Debugging wrapper
        def debug_torch_device_replacement(device_spec):
            print(f"  Before call - CPU override: {DeviceManager._cpu_override}")
            result = DeviceManager.torch_device_replacement(device_spec)
            print(f"  After call - CPU override: {DeviceManager._cpu_override}")
            print(f"  Requested '{device_spec}', got: {result}")
            return result
            
        # Patch the DeviceManager.torch_device_replacement temporarily for debugging
        original_func = DeviceManager.torch_device_replacement
        DeviceManager.torch_device_replacement = debug_torch_device_replacement
        
        try:
            # Now test the tensor movement
            print("Moving tensor to MPS:")
            mps_tensor = cpu_tensor.to('mps')
            print(f"Tensor device after move: {mps_tensor.device}")
            
            # Try with CPU
            print("\nMoving tensor to CPU:")
            cpu_tensor_again = mps_tensor.to('cpu')
            print(f"Tensor device after move: {cpu_tensor_again.device}")
            
            # Force reset the override flag if it's on
            if DeviceManager._cpu_override:
                print("\nResetting CPU override flag:")
                DeviceManager._cpu_override = False
                print(f"  After reset - CPU override: {DeviceManager._cpu_override}")
                
                # Try MPS again after reset
                print("Moving tensor to MPS after reset:")
                mps_tensor_after = cpu_tensor.to('mps')
                print(f"Tensor device after reset and move: {mps_tensor_after.device}")
                
        finally:
            # Restore original function
            DeviceManager.torch_device_replacement = original_func


if __name__ == '__main__':
    unittest.main()
