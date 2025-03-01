#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TorchDevice TTS Integration.
This module provides a PR-ready patch for TorchDevice to work with TTS.
"""

import os
import sys
import torch
import logging
import functools
import inspect
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the TorchDevice module to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import TorchDevice
import TorchDevice

def patch_torchdevice_for_tts():
    """
    Patch TorchDevice to work with TTS.
    This function adds the necessary patches to the TorchDevice module
    to make it compatible with TTS.
    """
    logger.info("Patching TorchDevice for TTS compatibility")
    
    # Store the original torch.serialization._get_restore_location function
    _original_get_restore_location = torch.serialization._get_restore_location
    
    # Create a patched version of torch.serialization._get_restore_location
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
    
    # Now patch torch.serialization directly
    torch.serialization._get_restore_location = patched_get_restore_location
    
    # Also patch the original torch.serialization module to fix isinstance checks
    original_code = """
    elif isinstance(map_location, torch.device):
    """
    patched_code = """
    elif hasattr(map_location, 'type') and hasattr(map_location, 'index'):
    """
    
    # Get the source code of torch.serialization._get_restore_location
    source = inspect.getsource(torch.serialization._get_restore_location)
    
    # Check if we need to patch the isinstance check
    if original_code.strip() in source:
        # Create a new version of the function with the isinstance check fixed
        source = source.replace(original_code, patched_code)
        
        # Create a namespace to execute the patched code
        namespace = {}
        exec(source, torch.serialization.__dict__, namespace)
        
        # Replace the original function with our patched version
        torch.serialization._get_restore_location = namespace['_get_restore_location']
        
        logger.info("Patched isinstance check in torch.serialization._get_restore_location")
    
    # Add the patch to TorchDevice's apply_patches method
    original_apply_patches = TorchDevice.TorchDevice.apply_patches
    
    @classmethod
    @functools.wraps(original_apply_patches)
    def patched_apply_patches(cls):
        """
        Patched version of apply_patches that also patches torch.serialization
        for TTS compatibility.
        """
        # Call the original apply_patches method
        original_apply_patches()
        
        # Patch torch.serialization._get_restore_location
        torch.serialization._get_restore_location = patched_get_restore_location
        
        logger.info("TorchDevice TTS compatibility patches applied")
    
    # Apply the patch to TorchDevice
    TorchDevice.TorchDevice.apply_patches = patched_apply_patches
    
    # Reapply patches to ensure our new patch is applied
    TorchDevice.TorchDevice.apply_patches()
    
    logger.info("TorchDevice patched for TTS compatibility")

def create_pr_patch():
    """
    Create a PR-ready patch for TorchDevice to add TTS compatibility.
    This function generates a diff that can be applied to the TorchDevice module.
    """
    # Get the source code of the patched_get_restore_location function
    source_code = inspect.getsource(patch_torchdevice_for_tts)
    
    # Create the PR patch
    pr_patch = """
# Add the following code to TorchDevice.py to add TTS compatibility

# In the imports section, add:
import functools
import inspect

# Add the following method to the TorchDevice class:
@classmethod
def patch_for_tts(cls):
    \"\"\"
    Patch TorchDevice to work with TTS.
    This method adds the necessary patches to make TorchDevice compatible with TTS.
    \"\"\"
    # Store the original _get_restore_location function
    _original_get_restore_location = torch.serialization._get_restore_location
    
    # Create a patched version
    @functools.wraps(_original_get_restore_location)
    def patched_get_restore_location(map_location):
        \"\"\"
        A patched version of torch.serialization._get_restore_location that handles
        TorchDevice device objects correctly.
        \"\"\"
        if cls.debug:
            logger.debug("Patched _get_restore_location called with map_location: %s", map_location)
        
        # Handle device objects by checking for attributes instead of using isinstance
        if hasattr(map_location, 'type') and hasattr(map_location, 'index'):
            if cls.debug:
                logger.debug("Detected device object with type: %s", map_location.type)
            
            # Create a restore function that handles device mapping
            def restore_location(storage, location):
                if cls.debug:
                    logger.debug("Restore location called with location: %s", location)
                # Just return the storage as is - we'll let TorchDevice handle device mapping
                return storage
            
            return restore_location
        
        # Fall back to original function for other cases
        return _original_get_restore_location(map_location)
    
    # Patch torch.serialization._get_restore_location
    torch.serialization._get_restore_location = patched_get_restore_location
    
    # Also patch the original torch.serialization module to fix isinstance checks
    original_code = \"\"\"
    elif isinstance(map_location, torch.device):
    \"\"\"
    patched_code = \"\"\"
    elif hasattr(map_location, 'type') and hasattr(map_location, 'index'):
    \"\"\"
    
    # Get the source code of torch.serialization._get_restore_location
    source = inspect.getsource(torch.serialization._get_restore_location)
    
    # Check if we need to patch the isinstance check
    if original_code.strip() in source:
        # Create a new version of the function with the isinstance check fixed
        source = source.replace(original_code, patched_code)
        
        # Create a namespace to execute the patched code
        namespace = {}
        exec(source, torch.serialization.__dict__, namespace)
        
        # Replace the original function with our patched version
        torch.serialization._get_restore_location = namespace['_get_restore_location']
        
        if cls.debug:
            logger.info("Patched isinstance check in torch.serialization._get_restore_location")
    
    if cls.debug:
        logger.info("TorchDevice TTS compatibility patches applied")

# Modify the apply_patches method to call patch_for_tts:
@classmethod
def apply_patches(cls):
    # ... existing code ...
    
    # Add TTS compatibility patches
    cls.patch_for_tts()
"""
    
    # Write the PR patch to a file
    patch_file = "torchdevice_tts_pr_patch.txt"
    with open(patch_file, "w") as f:
        f.write(pr_patch)
    
    logger.info("PR patch created and saved to %s", patch_file)
    
    return patch_file

def main():
    """Main function."""
    # Patch TorchDevice for TTS compatibility
    patch_torchdevice_for_tts()
    
    # Create PR patch
    pr_patch_file = create_pr_patch()
    
    # Print instructions
    print("\nTorchDevice has been patched for TTS compatibility.")
    print("A PR-ready patch has been created in:", pr_patch_file)
    print("\nTo make this patch permanent, you can:")
    print("1. Apply the changes in the patch file to TorchDevice.py")
    print("2. Submit a pull request to the TorchDevice repository")
    print("\nFor now, you can use the patched TorchDevice with TTS by importing this module before importing TTS.")

if __name__ == "__main__":
    main()
