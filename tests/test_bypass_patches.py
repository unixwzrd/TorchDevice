#!/usr/bin/env python
"""
Test bypass patches for functions that require CPU tensors.
"""

import unittest
import torch
import TorchDevice


class TestBypassPatches(unittest.TestCase):
    """Test that bypass patches correctly move tensors to CPU for specific functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Ensure TorchDevice is patched
        TorchDevice.core.patch.ensure_patched()
        
        # Create a tensor on the default device (likely MPS on this system)
        self.test_tensor = torch.tensor([1, 2, 3, 4, 5])
        self.lengths_tensor = torch.tensor([5])
        
    def test_pack_padded_sequence_bypass(self):
        """Test that pack_padded_sequence bypass works correctly."""
        # Create input tensors - batch_size=1, seq_len=5, features=3
        input_tensor = torch.randn(1, 5, 3)  # batch_first=True: (batch, seq_len, features)
        lengths = torch.tensor([5])  # Length of the sequence
        
        # This should work without error because the bypass patch moves tensors to CPU
        try:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                input_tensor, 
                lengths, 
                batch_first=True
            )
            self.assertIsNotNone(packed)
            print("✅ pack_padded_sequence worked correctly")
        except Exception as e:
            self.fail(f"pack_padded_sequence failed: {e}")
    
    def test_pad_packed_sequence_bypass(self):
        """Test that pad_packed_sequence bypass works correctly."""
        # Create input tensors - batch_size=1, seq_len=5, features=3
        input_tensor = torch.randn(1, 5, 3)  # batch_first=True: (batch, seq_len, features)
        lengths = torch.tensor([5])  # Length of the sequence
        
        # Pack first
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            input_tensor, 
            lengths, 
            batch_first=True
        )
        
        # Then unpack - this should also work due to bypass
        try:
            unpacked, lengths_unpacked = torch.nn.utils.rnn.pad_packed_sequence(
                packed, 
                batch_first=True
            )
            self.assertIsNotNone(unpacked)
            self.assertIsNotNone(lengths_unpacked)
            print("✅ pad_packed_sequence worked correctly")
        except Exception as e:
            self.fail(f"pad_packed_sequence failed: {e}")
    
    def test_pack_sequence_bypass(self):
        """Test that pack_sequence bypass works correctly."""
        # Create a list of tensors with different lengths (sorted in decreasing order)
        sequences = [
            torch.randn(4, 2),  # 4 timesteps, 2 features
            torch.randn(3, 2),  # 3 timesteps, 2 features
            torch.randn(2, 2),  # 2 timesteps, 2 features
        ]
        
        try:
            packed = torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted=False)
            self.assertIsNotNone(packed)
            print("✅ pack_sequence worked correctly")
        except Exception as e:
            self.fail(f"pack_sequence failed: {e}")
    
    def test_pad_sequence_bypass(self):
        """Test that pad_sequence bypass works correctly."""
        # Create a list of tensors with different lengths
        sequences = [
            torch.randn(3, 2),  # 3 timesteps, 2 features
            torch.randn(2, 2),  # 2 timesteps, 2 features
            torch.randn(4, 2),  # 4 timesteps, 2 features
        ]
        
        try:
            padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            self.assertIsNotNone(padded)
            print("✅ pad_sequence worked correctly")
        except Exception as e:
            self.fail(f"pad_sequence failed: {e}")


if __name__ == '__main__':
    unittest.main() 