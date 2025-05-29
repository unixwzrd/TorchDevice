"""
Test Module Import Structure
-------------------------
Verify that all TorchDevice modules import correctly and in the right order.
"""

import pytest
import sys
import logging
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_core_imports():
    """Test that core modules import correctly."""
    from TorchDevice.core import device, patch, logger
    assert device is not None
    assert patch is not None
    assert logger is not None

def test_ops_memory_imports():
    """Test that memory operation modules import correctly."""
    from TorchDevice.ops.memory import management, stats
    assert management is not None
    assert stats is not None

def test_ops_nn_imports():
    """Test that neural network modules import correctly."""
    from TorchDevice.ops.nn import (
        containers, layers, normalization,
        activation, attention, init
    )
    assert containers is not None
    assert layers is not None
    assert normalization is not None
    assert activation is not None
    assert attention is not None
    assert init is not None

def test_ops_streams_imports():
    """Test that stream modules import correctly."""
    from TorchDevice.ops.streams import cuda, mps, synchronize
    assert cuda is not None
    assert mps is not None
    assert synchronize is not None

def test_ops_autograd_imports():
    """Test that autograd modules import correctly."""
    from TorchDevice.ops.autograd import function, variable, grad_mode
    assert function is not None
    assert variable is not None
    assert grad_mode is not None

def test_ops_events_imports():
    """Test that event modules import correctly."""
    from TorchDevice.ops.events import cuda_events, mps_events, synchronize
    assert cuda_events is not None
    assert mps_events is not None
    assert synchronize is not None

def test_utils_imports():
    """Test that utility modules import correctly."""
    from TorchDevice.utils import (
        compile, device_utils, error_handling,
        profiling, type_utils
    )
    assert compile is not None
    assert device_utils is not None
    assert error_handling is not None
    assert profiling is not None
    assert type_utils is not None

def test_full_import():
    """Test importing the entire package."""
    import TorchDevice
    assert TorchDevice is not None

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 