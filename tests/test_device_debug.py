#!/usr/bin/env python
"""
Debug test to understand device detection and bypass behavior.
"""

import unittest
import torch
import TorchDevice
from tests.common.testing_utils import PrefixedTestCase


class TestDeviceDebug(PrefixedTestCase):
    """Test to debug device detection issues."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Ensure TorchDevice is patched
        TorchDevice.core.patch.ensure_patched()
    
    def test_device_detection_state(self):
        """Test the current state of device detection."""
        self.info("=== Device Detection State ===")
        self.info("Default device type: %s", TorchDevice.core.device.DeviceManager._default_device_type)
        self.info("CPU override active: %s", TorchDevice.core.device.DeviceManager.cpu_override())
        self.info("MPS available: %s", TorchDevice.core.device.DeviceManager.is_mps_available())
        
        # Test torch.device() with no arguments
        device = torch.device()
        self.info("torch.device() returns: %s", device)
        
        # Test torch.zeros without device
        zeros = torch.zeros(2, 3)
        self.info("torch.zeros(2, 3) device: %s", zeros.device)
        
        # Verify we're getting the expected device
        self.assertEqual(device.type, 'mps', "Expected MPS device, got %s" % device.type)
        self.assertEqual(zeros.device.type, 'mps', "Expected MPS device for zeros, got %s" % zeros.device.type)
    
    def test_bypass_context(self):
        """Test device behavior within bypass context."""
        self.info("=== Testing Bypass Context ===")
        from TorchDevice.core.config import bypass_argument_processing, is_bypass_active
        
        # Test outside bypass
        self.info("Bypass active (outside): %s", is_bypass_active())
        device_outside = torch.device()
        self.info("torch.device() outside bypass: %s", device_outside)
        
        # Test inside bypass
        with bypass_argument_processing():
            self.info("Bypass active (inside): %s", is_bypass_active())
            device_inside = torch.device()
            self.info("torch.device() inside bypass: %s", device_inside)
            
            zeros_inside = torch.zeros(2, 3)
            self.info("torch.zeros(2, 3) inside bypass device: %s", zeros_inside.device)
        
        # Test after bypass
        self.info("Bypass active (after): %s", is_bypass_active())
        device_after = torch.device()
        self.info("torch.device() after bypass: %s", device_after)
        
        # Verify bypass doesn't affect device detection
        self.assertEqual(device_outside.type, 'mps', "Device should be MPS outside bypass")
        self.assertEqual(device_after.type, 'mps', "Device should be MPS after bypass")
    
    def test_rnn_function_behavior(self):
        """Test what happens during and after RNN function calls."""
        self.info("=== Testing RNN Function Behavior ===")
        
        # Test pack_padded_sequence
        input_tensor = torch.randn(1, 5, 3)
        lengths = torch.tensor([5])
        
        self.info("Input tensor device: %s", input_tensor.device)
        self.info("Lengths tensor device: %s", lengths.device)
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_tensor, lengths, batch_first=True)
        self.info("packed type: %s", type(packed))
        
        if hasattr(packed, 'data'):
            self.info("packed.data device: %s", packed.data.device)
        elif isinstance(packed, tuple):
            self.info("packed is a tuple with %d elements", len(packed))
            for i, item in enumerate(packed):
                if hasattr(item, 'device'):
                    self.info("packed[%d] device: %s", i, item.device)
                else:
                    self.info("packed[%d] type: %s", i, type(item))
        else:
            self.info("packed is neither PackedSequence nor tuple: %s", type(packed))
        
        # Test what happens after pack_padded_sequence
        self.info("After pack_padded_sequence, torch.device() returns: %s", torch.device())
        zeros_after = torch.zeros(2, 3)
        self.info("After pack_padded_sequence, torch.zeros(2, 3) device: %s", zeros_after.device)
        
        # Verify the packed sequence is back on the correct device
        if hasattr(packed, 'data'):
            self.assertEqual(packed.data.device.type, 'mps',
                             "Packed sequence should be back on MPS, got %s" % packed.data.device.type)
        elif isinstance(packed, tuple) and len(packed) > 0:
            # If it's a tuple, check the first element that has a device
            for item in packed:
                if hasattr(item, 'device'):
                    self.assertEqual(item.device.type, 'mps',
                                     "Packed sequence element should be back on MPS, got %s" % item.device.type)
                    break
        
        self.assertEqual(zeros_after.device.type, 'mps',
                         "Zeros after RNN should be on MPS, got %s" % zeros_after.device.type)

    def test_pack_padded_sequence_return_type(self):
        """Test what pack_padded_sequence actually returns."""
        self.info("=== Testing pack_padded_sequence Return Type ===")
        
        # Create simple test data
        input_tensor = torch.randn(1, 3, 2)  # batch_size=1, seq_len=3, features=2
        lengths = torch.tensor([3])
        
        self.info("Input tensor shape: %s, device: %s", input_tensor.shape, input_tensor.device)
        self.info("Lengths tensor: %s, device: %s", lengths, lengths.device)
        
        try:
            # Call pack_padded_sequence directly
            packed = torch.nn.utils.rnn.pack_padded_sequence(input_tensor, lengths, batch_first=True)
            self.info("pack_padded_sequence returned: %s", type(packed))
            
            if hasattr(packed, 'data'):
                self.info("packed.data shape: %s, device: %s", packed.data.shape, packed.data.device)
                self.info("packed.batch_sizes: %s", packed.batch_sizes)
                self.info("packed.sorted_indices: %s", packed.sorted_indices)
                self.info("packed.unsorted_indices: %s", packed.unsorted_indices)
            elif isinstance(packed, tuple):
                self.info("packed is a tuple with %d elements", len(packed))
                for i, item in enumerate(packed):
                    if hasattr(item, 'device'):
                        self.info("packed[%d] shape: %s, device: %s", i, item.shape, item.device)
                    else:
                        self.info("packed[%d] type: %s, value: %s", i, type(item), item)
            else:
                self.info("packed is neither PackedSequence nor tuple: %s", type(packed))
                
        except Exception as e:
            self.info("pack_padded_sequence failed with error: %s", e)
            import traceback
            self.info("Traceback: %s", traceback.format_exc())


if __name__ == '__main__':
    unittest.main() 