"""
TorchDevice Neural Network Function Patches
-----------------------------------------
This module centralizes all patches for torch.nn and torch.nn.functional operations
to ensure proper device handling across different hardware.

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
from typing import Optional, Union, Tuple
from ..modules.TDLogger import auto_log


# Store original functions
t_embedding = torch.embedding if hasattr(torch, 'embedding') else None
t_nn_functional_embedding = F.embedding
t_linear = F.linear if hasattr(F, 'linear') else None
t_layer_norm = F.layer_norm if hasattr(F, 'layer_norm') else None


def _ensure_tensor_device(tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
    """Ensure tensor is on the correct device."""
    return tensor.to(target_device) if tensor.device != target_device else tensor


def _ensure_tensor_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Ensure tensor has the correct dtype."""
    return tensor.to(dtype=dtype) if tensor.dtype != dtype else tensor


@auto_log()
def embedding_replacement(
        input: torch.Tensor,
        weight: torch.Tensor,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False) -> torch.Tensor:
    """
    Replacement for torch.embedding and torch.nn.functional.embedding that handles device redirection.
    Ensures weight and input tensors are on the same device and have correct types.
    
    Args:
        input: Input tensor containing indices into the embedding matrix
        weight: The embedding matrix with shape (num_embeddings, embedding_dim)
        padding_idx: If specified, the entries at padding_idx do not contribute to the gradient
        max_norm: If given, each embedding vector with norm larger than max_norm is renormalized
        norm_type: The p of the p-norm to compute for the max_norm option
        scale_grad_by_freq: If given, gradient of each embedding is scaled by its frequency
        sparse: If True, gradient w.r.t. weight will be a sparse tensor
    """
    # Ensure tensors are on same device
    weight = _ensure_tensor_device(weight, input.device)
    
    # Ensure input has correct dtype for embedding lookup
    input = _ensure_tensor_dtype(input, torch.long)
    
    # Handle max_norm separately to avoid type issues
    if max_norm is not None:
        with torch.no_grad():
            # Use clone to avoid modifying the original weight tensor
            weight = weight.clone()
            norms = weight.float().norm(norm_type, dim=-1)
            mask = norms > max_norm
            if mask.any():
                scale = (max_norm / norms[mask]).unsqueeze(-1)
                weight[mask] = weight[mask] * scale.to(weight.dtype)
    
    # Call original embedding function without max_norm
    if t_nn_functional_embedding is not None:
        return t_nn_functional_embedding(
            input, weight, padding_idx, None, norm_type, scale_grad_by_freq, sparse
        )


@auto_log()
def linear_replacement(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Replacement for torch.nn.functional.linear that handles device redirection.
    Ensures all tensors are on the same device and have compatible dtypes.
    
    Args:
        input: Input tensor of shape (*, in_features)
        weight: Weight matrix of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features)
    """
    # Ensure tensors are on same device
    weight = _ensure_tensor_device(weight, input.device)
    if bias is not None:
        bias = _ensure_tensor_device(bias, input.device)
    
    # Ensure compatible dtypes
    if weight.dtype != input.dtype:
        weight = weight.to(dtype=input.dtype)
    if bias is not None and bias.dtype != input.dtype:
        bias = bias.to(dtype=input.dtype)
    
    if t_linear is not None:
        return t_linear(input, weight, bias)


@auto_log()
def layer_norm_replacement(
        input: torch.Tensor,
        normalized_shape: Union[int, Tuple[int, ...]],
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5) -> torch.Tensor:
    """
    Replacement for torch.nn.functional.layer_norm that handles device redirection.
    Ensures all tensors are on the same device and have compatible dtypes.
    
    Args:
        input: Input tensor of shape (N, *)
        normalized_shape: Input shape from an expected input of size
        weight: Optional weight tensor of shape (normalized_shape)
        bias: Optional bias tensor of shape (normalized_shape)
        eps: Small value added to denominator for numerical stability
    """
    # Ensure tensors are on same device and have compatible dtypes
    if weight is not None:
        weight = _ensure_tensor_device(weight, input.device)
        weight = _ensure_tensor_dtype(weight, input.dtype)
    if bias is not None:
        bias = _ensure_tensor_device(bias, input.device)
        bias = _ensure_tensor_dtype(bias, input.dtype)
    
    if t_layer_norm is not None:
        return t_layer_norm(input, normalized_shape, weight, bias, eps)


def apply_patches() -> None:
    """Apply all neural network related patches."""
    # Patch embedding functions
    if hasattr(torch, 'embedding'):
        torch.embedding = embedding_replacement
    if hasattr(torch.nn.functional, 'embedding'):
        torch.nn.functional.embedding = embedding_replacement
    
    # Patch linear operation
    if hasattr(torch.nn.functional, 'linear'):
        torch.nn.functional.linear = linear_replacement
    
    # Patch layer normalization
    if hasattr(torch.nn.functional, 'layer_norm'):
        torch.nn.functional.layer_norm = layer_norm_replacement
