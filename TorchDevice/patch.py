# This module is now redundant: patching is handled automatically in TorchDevice/__init__.py
# Kept for future internal use or as a placeholder for additional patch logic if needed.

from TorchDevice.cuda.streams import apply_patches as _apply_streams_patches
 
def _patch_all() -> None:
    """Internal: Apply all TorchDevice monkey-patches. Not for public use."""
    _apply_streams_patches()
    # Add additional patch calls here as you modularize more functionality 