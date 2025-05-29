"""
TorchDevice Neural Network Function Patches
-----------------------------------------
This module centralizes all patches for torch.nn and torch.nn.functional operations
to ensure proper device and type handling across different hardware.

Key areas covered:
- Embedding operations
- Linear/Dense operations
- Attention mechanisms
- Normalization layers
- Activation functions
- Dropout layers
"""

import torch
import torch.nn.functional as F
from typing import Optional, Sequence
from ..modules.TDLogger import auto_log

# Store original functions
t_nn_embedding = torch.nn.Embedding
t_nn_functional_embedding = F.embedding
t_nn_linear = torch.nn.Linear
t_nn_functional_linear = F.linear
t_nn_layer_norm = torch.nn.LayerNorm
t_nn_functional_layer_norm = F.layer_norm


def _ensure_tensor_device(tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
    """Ensure tensor is on the correct device."""
    return tensor.to(target_device) if tensor.device != target_device else tensor


def _ensure_tensor_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Ensure tensor has the correct dtype."""
    return tensor.to(dtype=dtype) if tensor.dtype != dtype else tensor


@auto_log()
def t_nn_embedding_forward(
        self,
        input: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for nn.Embedding that handles device redirection.
    """
    weight = self.weight
    padding_idx = self.padding_idx
    max_norm = self.max_norm
    norm_type = self.norm_type
    scale_grad_by_freq = self.scale_grad_by_freq
    sparse = self.sparse

    # Ensure tensors are on same device
    weight = _ensure_tensor_device(weight, input.device)
    
    # Ensure input has correct dtype for embedding lookup
    input = _ensure_tensor_dtype(input, torch.long)
    
    # Handle max_norm separately to avoid type issues
    if max_norm is not None:
        with torch.no_grad():
            norms = weight.norm(norm_type, dim=-1)
            mask = norms > max_norm
            if mask.any():
                scale = (max_norm / norms[mask]).unsqueeze(-1)
                weight = weight.clone()
                weight[mask] = weight[mask] * scale.to(weight.dtype)
    
    return t_nn_functional_embedding(
        input, weight, padding_idx, None, norm_type, scale_grad_by_freq, sparse
    )


@auto_log()
def t_nn_linear_forward(
        self,
        input: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for nn.Linear that handles device redirection.
    """
    weight = self.weight
    bias = self.bias

    # Ensure tensors are on same device
    weight = _ensure_tensor_device(weight, input.device)
    if bias is not None:
        bias = _ensure_tensor_device(bias, input.device)
    
    # Ensure compatible dtypes
    if weight.dtype != input.dtype:
        weight = weight.to(dtype=input.dtype)
    if bias is not None and bias.dtype != input.dtype:
        bias = bias.to(dtype=input.dtype)
    
    return t_nn_functional_linear(input, weight, bias)


@auto_log()
def t_nn_layer_norm_forward(
        self,
        input: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for nn.LayerNorm that handles device redirection.
    """
    weight = self.weight
    bias = self.bias
    normalized_shape = self.normalized_shape
    eps = self.eps

    # Ensure tensors are on same device and have compatible dtypes
    if weight is not None:
        weight = _ensure_tensor_device(weight, input.device)
        weight = _ensure_tensor_dtype(weight, input.dtype)
    if bias is not None:
        bias = _ensure_tensor_device(bias, input.device)
        bias = _ensure_tensor_dtype(bias, input.dtype)
    
    return t_nn_functional_layer_norm(input, normalized_shape, weight, bias, eps)


def apply_patches() -> None:
    """Apply all neural network related patches."""
    # Patch nn.Embedding
    torch.nn.Embedding.forward = t_nn_embedding_forward
    
    # Patch nn.Linear
    torch.nn.Linear.forward = t_nn_linear_forward
    
    # Patch nn.LayerNorm
    torch.nn.LayerNorm.forward = t_nn_layer_norm_forward
    
    # Patch functional interface
    torch.nn.functional.embedding = t_nn_functional_embedding
    torch.nn.functional.linear = t_nn_functional_linear
    torch.nn.functional.layer_norm = t_nn_functional_layer_norm
