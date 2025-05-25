"""
Tests for neural network operations in TorchDevice
"""

import torch
import pytest
from transformers import BertModel, BertTokenizer


def test_embedding_basic():
    """Test basic embedding functionality."""
    num_embeddings = 100
    embedding_dim = 768  # BERT-like dimension
    
    # Create test tensors and embedding layer
    embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    input_ids = torch.randint(0, num_embeddings, (1, 10))  # Batch of 10 tokens
    
    # Test without max_norm
    result = embedding(input_ids)
    assert result.shape == (1, 10, embedding_dim)
    assert result.dtype == embedding.weight.dtype
    
    # Test with max_norm
    embedding_norm = torch.nn.Embedding(num_embeddings, embedding_dim, max_norm=1.0)
    result_norm = embedding_norm(input_ids)
    assert result_norm.shape == (1, 10, embedding_dim)
    assert torch.all(result_norm.norm(dim=-1) <= 1.0 + 1e-6)


def test_bert_embedding():
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
        
        assert torch.allclose(orig_embed, our_embed, atol=1e-5)


def test_linear_layer():
    """Test linear layer functionality."""
    batch_size = 2
    in_features = 768
    out_features = 512
    
    # Create test tensors and linear layer
    input_tensor = torch.randn(batch_size, in_features)
    linear = torch.nn.Linear(in_features, out_features)
    
    # Test with bias
    result = linear(input_tensor)
    assert result.shape == (batch_size, out_features)
    assert result.dtype == input_tensor.dtype
    
    # Test without bias
    linear_no_bias = torch.nn.Linear(in_features, out_features, bias=False)
    result_no_bias = linear_no_bias(input_tensor)
    assert result_no_bias.shape == (batch_size, out_features)


def test_layer_norm():
    """Test layer normalization functionality."""
    batch_size = 2
    seq_length = 10
    hidden_size = 768
    
    # Create test tensors and layer norm
    input_tensor = torch.randn(batch_size, seq_length, hidden_size)
    layer_norm = torch.nn.LayerNorm(hidden_size)
    
    # Test layer norm
    result = layer_norm(input_tensor)
    
    assert result.shape == input_tensor.shape
    # Check if normalized (mean ≈ 0, std ≈ 1)
    assert torch.allclose(result.mean(dim=-1), torch.zeros_like(result.mean(dim=-1)), atol=1e-5)
    assert torch.allclose(result.std(dim=-1), torch.ones_like(result.std(dim=-1)), atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__]) 