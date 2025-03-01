#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS Patch for TorchDevice compatibility.
This module provides patches to make TTS work with TorchDevice.
"""

import torch
import logging
import functools
import builtins

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store original functions
_original_torch_device = torch.device
_original_isinstance = builtins.isinstance

# Create a patched version of torch.device
class PatchedDevice:
    """A patched version of torch.device that works with TorchDevice."""
    
    def __new__(cls, *args, **kwargs):
        logger.info(f"PatchedDevice.__new__ called with args: {args}, kwargs: {kwargs}")
        return _original_torch_device(*args, **kwargs)

# Create a patched version of isinstance
def patched_isinstance(obj, class_or_tuple):
    """A patched version of isinstance that handles torch.device properly."""
    logger.info(f"patched_isinstance called with obj: {type(obj)}, class_or_tuple: {class_or_tuple}")
    
    # Handle torch.device specifically
    if hasattr(obj, 'type') and hasattr(obj, 'index'):
        if class_or_tuple == torch.device:
            return True
        elif isinstance(class_or_tuple, tuple) and torch.device in class_or_tuple:
            return True
    
    # Fall back to original isinstance for other cases
    return _original_isinstance(obj, class_or_tuple)

def apply_patches():
    """Apply all patches needed for TTS to work with TorchDevice."""
    # Use global torch to avoid UnboundLocalError
    global torch
    
    logger.info("Applying TTS compatibility patches for TorchDevice")
    
    # Patch torch.device
    torch.device = PatchedDevice
    
    # Patch isinstance globally
    # This is a bit risky but necessary for compatibility
    builtins.isinstance = patched_isinstance
    
    # Patch torch.serialization._get_restore_location
    try:
        from torch.serialization import _get_restore_location
        
        # Store the original function
        _original_get_restore_location = _get_restore_location
        
        # Create a patched version
        @functools.wraps(_original_get_restore_location)
        def patched_get_restore_location(map_location):
            logger.info(f"patched_get_restore_location called with map_location: {map_location}")
            
            # Handle torch.device case specifically
            if hasattr(map_location, 'type') and hasattr(map_location, 'index'):
                # Create a restore function that handles device mapping
                def restore_location(storage, location):
                    logger.info(f"restore_location: storage={type(storage)}, location={location}")
                    if location.startswith('cuda'):
                        return storage.to('mps')
                    return storage
                
                return restore_location
            
            # Fall back to original function
            return _original_get_restore_location(map_location)
        
        # Apply the patch
        import torch.serialization
        torch.serialization._get_restore_location = patched_get_restore_location
        logger.info("Patched torch.serialization._get_restore_location")
        
    except Exception as e:
        logger.error(f"Failed to patch torch.serialization._get_restore_location: {e}")
    
    logger.info("TTS compatibility patches applied")

def remove_patches():
    """Remove all patches applied by apply_patches()."""
    # Use global torch to avoid UnboundLocalError
    global torch
    
    logger.info("Removing TTS compatibility patches")
    
    # Restore torch.device
    torch.device = _original_torch_device
    
    # Restore isinstance
    builtins.isinstance = _original_isinstance
    
    # Restore torch.serialization._get_restore_location
    try:
        import torch.serialization
        from torch.serialization import _get_restore_location
        if hasattr(_get_restore_location, '__wrapped__'):
            torch.serialization._get_restore_location = _get_restore_location.__wrapped__
            logger.info("Restored torch.serialization._get_restore_location")
    except Exception as e:
        logger.error(f"Failed to restore torch.serialization._get_restore_location: {e}")
    
    logger.info("TTS compatibility patches removed")
