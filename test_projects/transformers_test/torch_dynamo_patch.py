#!/usr/bin/env python
"""
Patch for torch._dynamo.device_interface to work with TorchDevice.

This patch ensures that the Event and Stream classes in torch._dynamo.device_interface
are properly recognized when using TorchDevice with MPS.
"""

import torch
import logging
import inspect
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)

def apply_torch_dynamo_patch():
    """
    Apply patch to torch._dynamo.device_interface to work with TorchDevice.
    
    This patch modifies the __new__ method of DeviceInterface to handle
    the case when Event and Stream are mocked by TorchDevice.
    """
    try:
        # Import the module we need to patch
        from torch._dynamo.device_interface import DeviceInterface
        
        # Store the original __new__ method
        original_new = DeviceInterface.__new__
        
        # Define our patched __new__ method
        def patched_new(cls, device_type, class_member):
            # Check if we're dealing with MPS device type
            if device_type == "mps":
                logger.info(f"Applying torch._dynamo.device_interface patch for MPS")
                
                # Skip the assertion checks for Event and Stream classes
                # when they are mocked by TorchDevice
                for key in ["Event", "Stream"]:
                    if key in class_member and not hasattr(class_member[key], "__mro__"):
                        logger.info(f"Skipping {key} check for MPS device")
                        # Create a dummy class that passes the issubclass check
                        from torch._dynamo.device_interface import _EventBase, _StreamBase
                        base_class = _EventBase if key == "Event" else _StreamBase
                        
                        class DummyClass(base_class):
                            def __init__(self, *args, **kwargs):
                                pass
                        
                        # Replace with our dummy class
                        class_member[key] = DummyClass
            
            # Call the original __new__ method
            return original_new(cls, device_type, class_member)
        
        # Apply our patch
        DeviceInterface.__new__ = patched_new
        logger.info("Successfully applied torch._dynamo.device_interface patch")
        
    except Exception as e:
        logger.error(f"Failed to apply torch._dynamo.device_interface patch: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Apply the patch
    apply_torch_dynamo_patch()
