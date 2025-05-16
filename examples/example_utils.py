"""
Utility functions for TorchDevice examples.

This module provides common utilities used across different example scripts.
"""
import random
import torch
import numpy as np


def set_deterministic_seed(seed=42):
    """
    Set deterministic seeds for reproducible examples.
    
    This sets seeds for Python's random module, NumPy, PyTorch CPU, 
    and PyTorch CUDA/MPS if available.
    
    Args:
        seed: The seed value to use (default: 42)
    """
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed (CPU and all devices)
    # Use standard PyTorch interfaces
    torch.manual_seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set MPS seed if available
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        if hasattr(torch.mps, 'manual_seed'):
            torch.mps.manual_seed(seed)
    
    # Extra deterministic settings for PyTorch
    if hasattr(torch.backends.cudnn, 'deterministic'):
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cudnn, 'benchmark'):
        torch.backends.cudnn.benchmark = False
    
    print(f"Set deterministic seed: {seed} for random, numpy, and torch") 