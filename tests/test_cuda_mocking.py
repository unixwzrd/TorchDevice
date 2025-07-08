import torch

# This import is all that's needed to trigger the patches
import TorchDevice
from tests.common.test_utils import PrefixedTestCase


class TestCudaMocking(PrefixedTestCase):
    """Test that CUDA functions are correctly mocked on MPS devices."""

    def test_cuda_is_mocked_on_mps(self):
        """On MPS, `is_available` and `is_built` should return True."""
        # This test is only relevant on systems with MPS hardware.
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            self.skipTest("Test requires an MPS device.")

        # After importing TorchDevice, these should be mocked to True
        self.assertTrue(
            torch.cuda.is_available(),
            "torch.cuda.is_available() should be mocked to return True on MPS"
        )
        self.assertTrue(
            torch.cuda.is_built(),
            "torch.cuda.is_built() should be mocked to return True on MPS"
        )
        self.assertEqual(
            torch.version.cuda,
            "11.8",
            f"torch.version.cuda should be mocked to '11.8', but got {torch.version.cuda}"
        )


if __name__ == '__main__':
    import unittest
    unittest.main()


