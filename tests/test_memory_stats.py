import unittest
import torch
import TorchDevice # Ensures patches are applied

class TestMemoryStats(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.mps_available = torch.backends.mps.is_available()
        # Importing TorchDevice at the top of the file should handle all necessary initializations.

    def test_mps_memory_stats_redirection(self):
        """Test redirection of CUDA memory stats to MPS stats when on MPS hardware."""
        if not self.mps_available:
            self.skipTest("MPS not available, skipping MPS-specific memory stats test.")

        # Ensure default device is MPS for these checks if applicable through TorchDevice
        # This assumes TorchDevice correctly sets default to MPS when available
        # and no override is active.

        # Test torch.cuda.memory_allocated()
        cuda_allocated = torch.cuda.memory_allocated()
        mps_allocated = torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else 0
        self.assertIsInstance(cuda_allocated, int)
        self.assertGreaterEqual(cuda_allocated, 0)
        self.assertEqual(cuda_allocated, mps_allocated, "torch.cuda.memory_allocated() should mirror torch.mps.current_allocated_memory()")

        # Test torch.cuda.memory_reserved()
        cuda_reserved = torch.cuda.memory_reserved()
        mps_reserved = torch.mps.driver_allocated_memory() if hasattr(torch.mps, 'driver_allocated_memory') else 0
        self.assertIsInstance(cuda_reserved, int)
        self.assertGreaterEqual(cuda_reserved, 0)
        self.assertEqual(cuda_reserved, mps_reserved, "torch.cuda.memory_reserved() should mirror torch.mps.driver_allocated_memory()")

        # Test torch.cuda.max_memory_allocated()
        cuda_max_allocated = torch.cuda.max_memory_allocated()
        # As per current implementation, max_allocated on MPS mirrors current_allocated
        self.assertIsInstance(cuda_max_allocated, int)
        self.assertGreaterEqual(cuda_max_allocated, 0)
        self.assertEqual(cuda_max_allocated, mps_allocated, "torch.cuda.max_memory_allocated() should mirror torch.mps.current_allocated_memory() on MPS")

        # Test torch.cuda.max_memory_reserved()
        cuda_max_reserved = torch.cuda.max_memory_reserved()
        # As per current implementation, max_reserved on MPS mirrors driver_allocated
        self.assertIsInstance(cuda_max_reserved, int)
        self.assertGreaterEqual(cuda_max_reserved, 0)
        self.assertEqual(cuda_max_reserved, mps_reserved, "torch.cuda.max_memory_reserved() should mirror torch.mps.driver_allocated_memory() on MPS")


    def test_mps_dynamic_memory_allocation(self):
        """Test that memory_allocated reflects dynamic tensor allocation on MPS."""
        if not self.mps_available:
            self.skipTest("MPS not available, skipping MPS dynamic allocation test.")

        initial_allocated = torch.cuda.memory_allocated()

        # Allocate a tensor on MPS device
        # The size should be significant enough to be noticeable by memory tracking
        # but not so large as to cause issues on all systems.
        # A 1024x1024 float32 tensor is 4MB.
        tensor_size = (1024, 1024)
        try:
            # Ensure the device context is MPS for this allocation
            # TorchDevice should handle redirection if code uses 'cuda' string here
            # but explicit 'mps' is clearer for testing MPS direct behavior.
            # However, we are testing the torch.cuda.memory_allocated patch,
            # so the *monitoring* is via cuda API, allocation can be via mps.
            device = torch.device('mps')
            test_tensor = torch.randn(tensor_size, device=device, dtype=torch.float32)
            
            current_allocated = torch.cuda.memory_allocated()

            # Check that allocated memory has increased
            # It's possible that due to caching or other factors, the increase might not be *exactly*
            # the tensor size, but it should be greater than initial.
            self.assertGreater(current_allocated, initial_allocated,
                               "Allocated memory should increase after tensor creation on MPS.")

            # Verify against direct MPS reporting if possible
            mps_current_allocated = torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else 0
            self.assertEqual(current_allocated, mps_current_allocated, 
                               "Patched cuda.memory_allocated should still match mps.current_allocated_memory after allocation.")

        finally:
            # Clean up the tensor and cache
            if 'test_tensor' in locals():
                del test_tensor
            if hasattr(torch.mps, 'empty_cache'): # Use direct mps call for cleanup in test
                 torch.mps.empty_cache()
            elif hasattr(torch.cuda, 'empty_cache'): # Fallback to our patched version
                 torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main()
