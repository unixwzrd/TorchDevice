"""
Test Module Import Structure
-------------------------
Verify that all TorchDevice modules import correctly and in the right order.
"""

import unittest
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

class TestImports(unittest.TestCase):
    def test_core_imports(self):
        """Test that core modules import correctly."""
        from TorchDevice.core import device, patch, logger
        self.assertIsNotNone(device)
        self.assertIsNotNone(patch)
        self.assertIsNotNone(logger)

    def test_ops_memory_imports(self):
        """Test that memory operation modules import correctly."""
        from TorchDevice.ops.memory import management, stats
        self.assertIsNotNone(management)
        self.assertIsNotNone(stats)

    def test_ops_nn_imports(self):
        """Test that neural network modules import correctly."""
        from TorchDevice.ops.nn import (
            containers, layers, normalization,
            activation, attention, init
        )
        self.assertIsNotNone(containers)
        self.assertIsNotNone(layers)
        self.assertIsNotNone(normalization)
        self.assertIsNotNone(activation)
        self.assertIsNotNone(attention)
        self.assertIsNotNone(init)

    def test_ops_streams_imports(self):
        """Test that stream modules import correctly."""
        from TorchDevice.ops.streams import cuda, mps, sync
        self.assertIsNotNone(cuda)
        self.assertIsNotNone(mps)
        self.assertIsNotNone(sync)

    def test_ops_autograd_imports(self):
        """Test that autograd modules import correctly."""
        from TorchDevice.ops.autograd import function, variable, grad_mode
        self.assertIsNotNone(function)
        self.assertIsNotNone(variable)
        self.assertIsNotNone(grad_mode)

    def test_ops_events_imports(self):
        """Test that event modules import correctly."""
        from TorchDevice.ops.events import cuda, mps, sync
        self.assertIsNotNone(cuda)
        self.assertIsNotNone(mps)
        self.assertIsNotNone(sync)

    def test_utils_imports(self):
        """Test that utility modules import correctly."""
        from TorchDevice.utils import (
            compile, device_utils, error_handling,
            profiling, type_utils
        )
        self.assertIsNotNone(compile)
        self.assertIsNotNone(device_utils)
        self.assertIsNotNone(error_handling)
        self.assertIsNotNone(profiling)
        self.assertIsNotNone(type_utils)

    def test_full_import(self):
        """Test importing the entire package."""
        import TorchDevice
        self.assertIsNotNone(TorchDevice)

if __name__ == '__main__':
    unittest.main() 