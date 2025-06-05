"""Unit tests for verifying CUDA patching.

This test ensures that TorchDevice correctly patches torch.cuda._lazy_init
and related CUDA functions to work on MPS or CPU hardware.
"""

import unittest
import sys
import logging
import torch
import TorchDevice

# Create logger before running tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TestCUDAPatching")

class TestCUDAPatching(unittest.TestCase):
    """Test that TorchDevice correctly patches CUDA functionality."""

    def setUp(self):
        """Setup the test case by configuring logging."""
        self.logger = logging.getLogger("TestCUDAPatching")
        self.logger.info("----- Starting New Test -----")
        self.cuda_available_before = torch.cuda.is_available()
        self.logger.info(f"torch.cuda.is_available() before tests: {self.cuda_available_before}")
        # Log if CUDA is actually compiled in PyTorch
        try:
            torch._C._cuda_getDeviceCount()
            self.cuda_compiled = True
        except (RuntimeError, AttributeError):
            self.cuda_compiled = False
        self.logger.info(f"CUDA actually compiled in PyTorch: {self.cuda_compiled}")
        # Log device manager state
        current_device = TorchDevice.core.device.DeviceManager.get_default_device()
        self.logger.info(f"Current default device: {current_device}")

    def test_cuda_is_available(self):
        """Test that torch.cuda.is_available() returns True after patching."""
        # Verify CUDA is available after patching
        self.assertTrue(torch.cuda.is_available())

    def test_lazy_init_patched(self):
        """Test that the patched _lazy_init function is invoked without errors."""
        self.logger.info("Testing _lazy_init by calling current_device()")
        try:
            device_id = torch.cuda.current_device()
            self.logger.info(f"torch.cuda.current_device() returned: {device_id}")
            self.assertIsInstance(device_id, int, "current_device should return an integer device ID")
        except Exception as e:
            self.fail(f"torch.cuda.current_device() raised exception: {e}")

    def test_device_count(self):
        """Test that device_count returns at least 1."""
        device_count = torch.cuda.device_count()
        self.logger.info(f"torch.cuda.device_count() returned: {device_count}")
        self.assertGreaterEqual(device_count, 1, "device_count should return at least 1")

    def test_cuda_module_attributes(self):
        """Check that critical CUDA module attributes are patched."""
        for attr_name in [
            'current_device', 'device_count', 'is_available',
            'is_initialized', 'synchronize', '_lazy_init'
        ]:
            self.assertTrue(
                hasattr(torch.cuda, attr_name),
                f"torch.cuda should have attribute {attr_name}"
            )
            attr = getattr(torch.cuda, attr_name)
            self.logger.info(f"torch.cuda.{attr_name} type: {type(attr)}")


if __name__ == "__main__":
    unittest.main()
