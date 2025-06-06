#!/usr/bin/env python
"""
Simple test file to verify tensor movement functions work correctly.
This isolates a few specific functions for testing.
"""
import unittest
import torch
import os
import sys

# Silence most logging to focus on functionality
os.environ['TORCHDEVICE_LOG_LEVEL'] = 'WARNING'

# Import TorchDevice to ensure patches are applied
import TorchDevice

class TestSingleFunction(unittest.TestCase):
    """Test a single function to isolate issues."""

    def setUp(self):
        """Set up test environment."""
        # Check if MPS is available
        self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Skip tests if MPS is not available
        if not self.mps_available:
            self.skipTest("MPS is not available on this system")
            
        # Using MPS device for testing
        self.device = torch.device('mps')
        print(f"Using device: {self.device}")

    def test_tensor_creation_and_movement(self):
        """Test basic tensor creation and movement to verify functions."""
        # Create tensor on CPU explicitly
        cpu_tensor = torch.ones(2, 3, device="cpu:-1")  # Force CPU with override
        print(f"cpu_tensor device: {cpu_tensor.device}")
        
        # Move to MPS
        mps_tensor = cpu_tensor.to(self.device)
        print(f"mps_tensor device: {mps_tensor.device}")
        self.assertEqual(mps_tensor.device.type, 'mps')
        
        # Move back to CPU
        cpu_tensor_again = mps_tensor.cpu()
        print(f"cpu_tensor_again device: {cpu_tensor_again.device}")
        self.assertEqual(cpu_tensor_again.device.type, 'cpu')
        
        # Check that values are preserved
        print(f"Comparing tensors on devices: {cpu_tensor.device} and {cpu_tensor_again.device}")
        # Ensure both tensors are on CPU for comparison
        if cpu_tensor.device.type != 'cpu':
            cpu_tensor = cpu_tensor.cpu()
        self.assertTrue(torch.allclose(cpu_tensor, cpu_tensor_again))
        
        # Test cuda() function which should redirect to MPS
        mps_tensor2 = cpu_tensor.cuda()
        print(f"mps_tensor2 device (from cuda call): {mps_tensor2.device}")
        self.assertEqual(mps_tensor2.device.type, 'mps')
        
        print("Tensor movement tests passed")

if __name__ == '__main__':
    unittest.main(argv=sys.argv[:1])
