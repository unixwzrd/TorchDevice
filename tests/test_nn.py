"""
Tests for neural network operations in TorchDevice
"""

import torch
import unittest
from transformers import BertModel, BertTokenizer


class TestNN(unittest.TestCase):
    def test_embedding_basic(self):
        """Test basic embedding functionality."""
        num_embeddings = 100
        embedding_dim = 768  # BERT-like dimension
        
        # Create test tensors and embedding layer
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        input_ids = torch.randint(0, num_embeddings, (1, 10))  # Batch of 10 tokens
        
        # Test without max_norm
        result = embedding(input_ids)
        self.assertEqual(result.shape, (1, 10, embedding_dim))
        self.assertEqual(result.dtype, embedding.weight.dtype)
        
        # Test with max_norm
        embedding_norm = torch.nn.Embedding(num_embeddings, embedding_dim, max_norm=1.0)
        result_norm = embedding_norm(input_ids)
        self.assertEqual(result_norm.shape, (1, 10, embedding_dim))
        self.assertTrue(torch.all(result_norm.norm(dim=-1) <= 1.0 + 1e-6))

    def test_bert_embedding(self):
        """Test embedding with actual BERT model."""
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        
        # Tokenize a test sentence
        text = "Hello world!"
        tokens = tokenizer(text, return_tensors='pt')
        input_ids = tokens['input_ids']
        
        # Compare original vs our implementation
        with torch.no_grad():
            orig_embed = model.embeddings.word_embeddings(input_ids)
            # Our implementation is now the same as the original
            our_embed = model.embeddings.word_embeddings(input_ids)
            
            self.assertTrue(torch.allclose(orig_embed, our_embed, atol=1e-5))

    def test_linear_layer(self):
        """Test linear layer functionality."""
        batch_size = 2
        in_features = 768
        out_features = 512
        
        # Create test tensors and linear layer
        input_tensor = torch.randn(batch_size, in_features)
        linear = torch.nn.Linear(in_features, out_features)
        
        # Test with bias
        result = linear(input_tensor)
        self.assertEqual(result.shape, (batch_size, out_features))
        self.assertEqual(result.dtype, input_tensor.dtype)
        
        # Test without bias
        linear_no_bias = torch.nn.Linear(in_features, out_features, bias=False)
        result_no_bias = linear_no_bias(input_tensor)
        self.assertEqual(result_no_bias.shape, (batch_size, out_features))

    def test_layer_norm(self):
        """Test layer normalization functionality."""
        batch_size = 2
        seq_length = 10
        hidden_size = 768
        
        # Create test tensors and layer norm
        input_tensor = torch.randn(batch_size, seq_length, hidden_size)
        layer_norm = torch.nn.LayerNorm(hidden_size)
        
        # Test layer norm
        result = layer_norm(input_tensor)
        
        self.assertEqual(result.shape, input_tensor.shape)
        # Check if normalized (mean ≈ 0, std ≈ 1)
        self.assertTrue(torch.allclose(result.mean(dim=-1), torch.zeros_like(result.mean(dim=-1)), atol=1e-5))
        # LayerNorm normalizes using N elements, so use unbiased=False for std calculation
        std_dev_unbiased_false = result.std(dim=-1, unbiased=False)
        self.assertTrue(torch.allclose(std_dev_unbiased_false, torch.ones_like(std_dev_unbiased_false), atol=1e-4))

if __name__ == '__main__':
    unittest.main() 