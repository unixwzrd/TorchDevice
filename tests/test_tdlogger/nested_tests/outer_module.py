"""
Outer module for testing nested function calls with TorchDevice.
"""

import torch
from .inner_module import inner_function


def outer_function():
    """Test function that calls inner_function."""
    # Create tensors and move them to the appropriate device
    x = torch.randn(3, 3)
    
    # Move tensor to device - will be intercepted by TorchDevice
    x = x.to('cuda')  
    
    # Call inner function
    result = inner_function(x)
    
    return result


def outer_wrapper():
    """Wrapper to test deeper call stack"""
    # Create a tensor and do some operations that will be intercepted
    device = torch.device('cuda')  # This will be logged even though we don't use the result
    
    # Check device properties - will be intercepted
    torch.cuda.get_device_properties(0)
    
    # Call the outer function
    return outer_function() 