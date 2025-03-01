#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TorchDevice TTS Patch.
This module provides patches for TorchDevice to work with TTS.
"""

import os
import sys
import torch
import logging
import functools
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store original functions for later restoration
_original_get_restore_location = None

def patch_torch_serialization():
    """
    Patch torch.serialization to handle TorchDevice objects correctly.
    This is necessary for TTS to work with TorchDevice.
    """
    global _original_get_restore_location
    
    # Store the original function
    _original_get_restore_location = torch.serialization._get_restore_location
    
    # Create a patched version
    @functools.wraps(_original_get_restore_location)
    def patched_get_restore_location(map_location):
        """
        A patched version of torch.serialization._get_restore_location that handles
        TorchDevice device objects correctly.
        """
        logger.debug("Patched _get_restore_location called with map_location: %s", map_location)
        
        # Handle device objects by checking for attributes instead of using isinstance
        if hasattr(map_location, 'type') and hasattr(map_location, 'index'):
            logger.debug("Detected device object with type: %s", map_location.type)
            
            # Create a restore function that handles device mapping
            def restore_location(storage, location):
                logger.debug("Restore location called with location: %s", location)
                # Just return the storage as is - we'll let TorchDevice handle device mapping
                return storage
            
            return restore_location
        
        # Fall back to original function for other cases
        return _original_get_restore_location(map_location)
    
    # Apply the patch
    torch.serialization._get_restore_location = patched_get_restore_location
    
    logger.info("TorchDevice TTS compatibility patches applied")

def restore_torch_serialization():
    """
    Restore the original torch.serialization functions.
    """
    global _original_get_restore_location
    
    if _original_get_restore_location is not None:
        torch.serialization._get_restore_location = _original_get_restore_location
        logger.info("TorchDevice TTS compatibility patches removed")

def apply_tts_patches():
    """
    Apply all patches needed for TTS to work with TorchDevice.
    """
    patch_torch_serialization()

def remove_tts_patches():
    """
    Remove all patches applied by apply_tts_patches().
    """
    restore_torch_serialization()

# Apply patches when this module is imported
apply_tts_patches()
