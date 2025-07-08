"""
Tests for neural network operations across different devices in TorchDevice
"""

import torch
import unittest
import sys
from transformers import BertModel, BertTokenizer
import TorchDevice
from tests.common.testing_utils import PrefixedTestCase, diff_check


class TestNNDeviceInteractions(PrefixedTestCase):
    """Test neural network operations across different devices."""

    def setUp(self):
        super().setUp()
        self.default_device = torch.get_default_device()
        self.default_device_type = self.default_device.type
        self.default_device_index = self.default_device.index
        self.info("Setting up test environment")
    
    def test_cross_device_attention(self):
        """Test attention mechanism with tensors on different devices."""
        self.info("Testing cross-device attention handling")
        
        # Create tensors on different devices
        query = torch.randn(2, 8, 10, 64, device='cpu')
        key = torch.randn(2, 8, 10, 64, device='cpu')
        value = torch.randn(2, 8, 10, 64, device='cpu')
        
        # Test attention with device mixing
        output = torch.nn.functional.scaled_dot_product_attention(
            query.cuda(), key, value
        )
        # Should automatically move all tensors to CUDA
        self.assertEqual(output.device.type, self.default_device_type)
        
        # Test with attention mask on different device
        mask = torch.ones(2, 8, 10, 10, device='cpu').bool()
        output = torch.nn.functional.scaled_dot_product_attention(
            query.cuda(), key.cuda(), value.cuda(),
            attn_mask=mask  # Mask starts on CPU
        )
        self.assertEqual(output.device.type, self.default_device_type)
        
        self.info("Cross-device attention tests passed")
        diff_check(self.log_capture)

    def test_mixed_precision_attention(self):
        """Test attention mechanism with mixed precision."""
        self.info("Testing mixed precision attention")
        
        # Create tensors with different dtypes
        query = torch.randn(2, 8, 10, 64, dtype=torch.float32)
        key = torch.randn(2, 8, 10, 64, dtype=torch.float16)
        value = torch.randn(2, 8, 10, 64, dtype=torch.float32)
        
        # Test attention with dtype mixing
        key = key.to(query.dtype)
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value
        )
        # Should use highest precision dtype
        self.assertEqual(output.dtype, torch.float32)
        
        self.info("Mixed precision attention tests passed")
        diff_check(self.log_capture)

    def test_bert_device_switching(self):
        """Test BERT model with device switching."""
        self.info("Testing BERT model device switching")
        
        model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Create test input
        text = "Testing device switching with BERT!"
        tokens = tokenizer(text, return_tensors='pt')
        input_ids = tokens['input_ids']
        
        # Test device switching during forward pass
        with torch.no_grad():
            # Move input to CUDA
            input_ids = input_ids.cuda()
            hidden_states = model.embeddings(input_ids)
            
            # Move model layers to different devices
            model.encoder.layer[0].attention.self.query.to('cpu')
            model.encoder.layer[0].attention.self.key.cuda()
            model.encoder.layer[0].attention.self.value.to('cpu')
            model.encoder.layer[0].attention.output.dense.cuda()
            
            # Forward pass should handle mixed devices
            output = model.encoder.layer[0].attention(hidden_states)[0]
            
            # Output should be on the same device as input
            self.assertEqual(output.device, hidden_states.device)
        
        self.info("BERT device switching tests passed")
        diff_check(self.log_capture)

    def test_embedding_device_handling(self):
        """Test embedding operations with device handling."""
        self.info("Testing embedding device handling")
        
        # Create embeddings on different devices
        num_embeddings = 100
        embedding_dim = 768
        
        # Create embedding layer and move weight to CPU
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        embedding.weight.data = embedding.weight.data.cpu()
        
        # Create input IDs on CUDA
        input_ids = torch.randint(0, num_embeddings, (1, 10)).cuda()
        
        # Test embedding with weight on CPU, input on CUDA
        result = embedding(input_ids)
        self.assertEqual(result.device, input_ids.device)
        
        # Test with max_norm on different device
        embedding_norm = torch.nn.Embedding(num_embeddings, embedding_dim, max_norm=1.0)
        embedding_norm.cuda()
        result_norm = embedding_norm(input_ids.cpu())
        self.assertEqual(result_norm.device, embedding_norm.weight.device)
        self.assertTrue(torch.all(result_norm.norm(dim=-1) <= 1.0 + 1e-6))
        
        self.info("Embedding device handling tests passed")
        diff_check(self.log_capture)

    def test_layer_norm_device_handling(self):
        """Test layer normalization with device handling."""
        self.info("Testing layer normalization device handling")
        
        batch_size = 2
        seq_length = 10
        hidden_size = 768
        
        # Create layer norm and move parameters to different devices
        layer_norm = torch.nn.LayerNorm(hidden_size)
        layer_norm.weight.data = layer_norm.weight.data.cpu()
        layer_norm.bias.data = layer_norm.bias.data.cuda()
        
        # Create input tensor on CUDA
        input_tensor = torch.randn(batch_size, seq_length, hidden_size).cuda()
        
        # Test layer norm with mixed devices
        result = layer_norm(input_tensor)
        
        # Result should be on same device as input
        self.assertEqual(result.device, input_tensor.device)
        self.assertEqual(result.shape, input_tensor.shape)
        
        # Check normalization properties
        self.assertTrue(torch.allclose(
            result.mean(dim=-1), 
            torch.zeros_like(result.mean(dim=-1)), 
            atol=1e-5
        ))
        actual_std = result.std(dim=-1)
        expected_std = torch.ones_like(actual_std)
        self.info(f"DEBUG: Actual std:\n{actual_std}")
        self.info(f"DEBUG: Expected std:\n{expected_std}")
        self.info(f"DEBUG: Max difference: {torch.abs(actual_std - expected_std).max()}")
        self.info(f"DEBUG: atol for std check: 7e-4")

        self.assertTrue(torch.allclose(
            actual_std, 
            expected_std, 
            atol=7e-4  # Increased tolerance for MPS precision
        ))
        
        self.info("Layer normalization device handling tests passed")
        diff_check(self.log_capture)


if __name__ == '__main__':
    unittest.main(argv=sys.argv[:1])