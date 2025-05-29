"""
TorchDevice Attention Mechanism Patches
-------------------------------------
This module handles patches for attention mechanisms, focusing on transformer architectures.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from ..modules.TDLogger import auto_log


# Store original functions if they exist
t_scaled_dot_product_attention = getattr(F, 'scaled_dot_product_attention', None)


@auto_log()
def scaled_dot_product_attention_replacement(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replacement for torch.nn.functional.scaled_dot_product_attention.
    Computes scaled dot product attention with optional masking.
    
    Args:
        query: Query tensor of shape (batch, num_heads, seq_len_q, head_dim)
        key: Key tensor of shape (batch, num_heads, seq_len_k, head_dim)
        value: Value tensor of shape (batch, num_heads, seq_len_v, head_dim)
        attn_mask: Optional mask tensor of shape (batch, num_heads, seq_len_q, seq_len_k)
        dropout_p: Dropout probability (0.0 means no dropout)
        is_causal: Whether to apply causal masking
        scale: Optional float to scale attention scores by. If None, uses 1/sqrt(head_dim)
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (attention output, attention weights)
    """
    # Ensure all tensors are on same device and have compatible dtypes
    key = key.to(device=query.device, dtype=query.dtype)
    value = value.to(device=query.device, dtype=query.dtype)
    
    if scale is None:
        scale = 1.0 / (query.size(-1) ** 0.5)
    
    # Compute attention scores
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply attention mask if provided
    if attn_mask is not None:
        attn_mask = attn_mask.to(device=query.device)
        attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))
    
    # Apply causal mask if requested
    if is_causal:
        seq_len = query.size(-2)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=query.device),
            diagonal=1
        ).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
    
    # Apply softmax and dropout
    attn_weights = F.softmax(attn_weights, dim=-1)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights, value)
    
    return attn_output, attn_weights


@auto_log()
def multi_head_attention_forward_replacement(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        embed_dim: int,
        num_heads: int,
        q_proj_weight: torch.Tensor,
        k_proj_weight: torch.Tensor,
        v_proj_weight: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        training: bool = True,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        is_causal: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Replacement for torch.nn.functional.multi_head_attention_forward.
    Implements multi-head attention with projections.
    
    Args:
        query: Query tensor of shape (seq_len_q, batch, embed_dim)
        key: Key tensor of shape (seq_len_k, batch, embed_dim)
        value: Value tensor of shape (seq_len_v, batch, embed_dim)
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        q_proj_weight: Query projection weight matrix
        k_proj_weight: Key projection weight matrix
        v_proj_weight: Value projection weight matrix
        out_proj_weight: Output projection weight matrix
        out_proj_bias: Optional output projection bias
        attn_mask: Optional mask tensor
        dropout_p: Dropout probability
        training: Whether in training mode
        key_padding_mask: Optional mask for key padding
        need_weights: If True, returns attention weights
        is_causal: Whether to apply causal masking
    
    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: (attention output, attention weights if need_weights)
    """
    # Project query, key, value
    head_dim = embed_dim // num_heads
    
    # Linear projections
    q = F.linear(query, q_proj_weight)
    k = F.linear(key, k_proj_weight)
    v = F.linear(value, v_proj_weight)
    
    # Reshape for multi-head attention
    q = q.contiguous().view(-1, query.size(1), num_heads, head_dim).transpose(0, 2)
    k = k.contiguous().view(-1, key.size(1), num_heads, head_dim).transpose(0, 2)
    v = v.contiguous().view(-1, value.size(1), num_heads, head_dim).transpose(0, 2)
    
    # Compute scaled dot-product attention
    attn_output, attn_weights = scaled_dot_product_attention_replacement(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=is_causal
    )
    
    # Reshape and apply output projection
    attn_output = attn_output.transpose(1, 2).contiguous().view(
        query.size(0), query.size(1), embed_dim
    )
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    
    if need_weights:
        return attn_output, attn_weights
    else:
        return attn_output, None


def apply_patches() -> None:
    """Apply all attention related patches."""
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention_replacement
    
    if hasattr(torch.nn.functional, 'multi_head_attention_forward'):
        torch.nn.functional.multi_head_attention_forward = multi_head_attention_forward_replacement 