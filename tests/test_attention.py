"""
Tests for attention mechanisms in TorchDevice
"""

import torch
import pytest
from transformers import BertModel
from TorchDevice.device.attention import (
    scaled_dot_product_attention_replacement,
    multi_head_attention_forward_replacement
)


def test_scaled_dot_product_attention():
    """Test basic scaled dot product attention."""
    batch_size = 2
    num_heads = 8
    seq_len = 10
    head_dim = 64
    
    # Create test tensors
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Test without mask
    output, weights = scaled_dot_product_attention_replacement(query, key, value)
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
    
    # Test with attention mask
    mask = torch.ones(batch_size, num_heads, seq_len, seq_len).bool()
    mask[:, :, :, -1] = False  # Mask out last position
    output_masked, weights_masked = scaled_dot_product_attention_replacement(
        query, key, value, attn_mask=mask
    )
    assert torch.all(weights_masked[:, :, :, -1] == 0)  # Last position should have zero attention
    
    # Test with dropout
    output_drop, _ = scaled_dot_product_attention_replacement(
        query, key, value, dropout_p=0.1
    )
    assert output_drop.shape == (batch_size, num_heads, seq_len, head_dim)
    
    # Test causal masking
    output_causal, weights_causal = scaled_dot_product_attention_replacement(
        query, key, value, is_causal=True
    )
    # Check that future positions are masked
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert torch.all(weights_causal[:, :, i, j] == 0)


def test_multi_head_attention():
    """Test multi-head attention mechanism."""
    seq_len = 10
    batch_size = 2
    embed_dim = 256
    num_heads = 8
    
    # Create test tensors
    query = torch.randn(seq_len, batch_size, embed_dim)
    key = torch.randn(seq_len, batch_size, embed_dim)
    value = torch.randn(seq_len, batch_size, embed_dim)
    
    # Create projection weights
    q_proj = torch.randn(embed_dim, embed_dim)
    k_proj = torch.randn(embed_dim, embed_dim)
    v_proj = torch.randn(embed_dim, embed_dim)
    out_proj = torch.randn(embed_dim, embed_dim)
    out_bias = torch.randn(embed_dim)
    
    # Test basic forward pass
    output, attn_weights = multi_head_attention_forward_replacement(
        query, key, value,
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_proj_weight=q_proj,
        k_proj_weight=k_proj,
        v_proj_weight=v_proj,
        out_proj_weight=out_proj,
        out_proj_bias=out_bias
    )
    
    assert output.shape == (seq_len, batch_size, embed_dim)
    assert attn_weights is not None
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    # Test without attention weights
    output_no_weights, attn_weights_none = multi_head_attention_forward_replacement(
        query, key, value,
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_proj_weight=q_proj,
        k_proj_weight=k_proj,
        v_proj_weight=v_proj,
        out_proj_weight=out_proj,
        out_proj_bias=out_bias,
        need_weights=False
    )
    
    assert output_no_weights.shape == (seq_len, batch_size, embed_dim)
    assert attn_weights_none is None


def test_bert_attention():
    """Test attention with BERT model."""
    model = BertModel.from_pretrained('bert-base-uncased')
    attention = model.encoder.layer[0].attention
    
    # Get attention parameters
    q_weight = attention.self.query.weight
    k_weight = attention.self.key.weight
    v_weight = attention.self.value.weight
    out_weight = attention.output.dense.weight
    out_bias = attention.output.dense.bias
    
    # Create test input
    seq_len = 10
    batch_size = 2
    hidden_size = 768
    num_heads = 12
    
    hidden_states = torch.randn(seq_len, batch_size, hidden_size)
    
    # Compare original vs our implementation
    with torch.no_grad():
        # Our implementation
        our_output, our_weights = multi_head_attention_forward_replacement(
            hidden_states, hidden_states, hidden_states,
            embed_dim=hidden_size,
            num_heads=num_heads,
            q_proj_weight=q_weight,
            k_proj_weight=k_weight,
            v_proj_weight=v_weight,
            out_proj_weight=out_weight,
            out_proj_bias=out_bias
        )
        
        # Original implementation (through BERT's attention layer)
        orig_output = attention(hidden_states)[0]
        
        # Check outputs are close
        assert torch.allclose(our_output, orig_output, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__]) 