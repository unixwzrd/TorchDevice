"""
Inner module for testing nested function calls with TorchDevice.
"""

import torch


def inner_function(x):
    """Test function that performs operations on tensors."""
    # Create another tensor and perform operations
    y = torch.randn(3, 3)
    y = y.cuda()  # This will be redirected if needed
    
    # Perform some operations
    z = x + y
    
    # Test device synchronization
    torch.cuda.synchronize()
    
    # Use z to avoid unused variable warning
    result = torch.sum(z)
    
    return result 