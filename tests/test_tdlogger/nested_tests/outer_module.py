"""
Outer module for testing nested function calls with TorchDevice.
"""

import torch
from TorchDevice import log
from .inner_module import inner_function


def outer_function():
    """Test function that calls inner_function."""
    # Create tensors and move them to the appropriate device
    x = torch.randn(3, 3)
    x = x.to('cuda')  # This will be redirected if needed
    
    # Call inner function
    result = inner_function(x)
    
    return result


def outer_wrapper():
    """Wrapper to test deeper call stack"""
    log_message("Called from outer_wrapper", "outer_wrapper")
    outer_function() 