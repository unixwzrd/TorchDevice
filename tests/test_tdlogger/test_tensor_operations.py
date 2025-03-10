"""
Test complex tensor operations with TorchDevice.

This module tests the logging of various tensor operations with TorchDevice,
including creation, movement between devices, and arithmetic operations.
"""
import logging
import unittest
from pathlib import Path

import torch
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture

import TorchDevice  # Import TorchDevice to ensure CUDA redirection is set up

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Ensure this configuration is applied
)
logger = logging.getLogger(__name__)


class TestTensorOperations(unittest.TestCase):
    """Test complex tensor operations with TorchDevice."""
    
    def setUp(self):
        """Set up logger capture for this test."""
        print("\n" + "=" * 80)
        print(f"Starting test: {self._testMethodName}")
        print("=" * 80)
        
        # Set up log capture with a unique test name.
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)        


    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        teardown_log_capture(self.log_capture)
        
        # Print a footer for the test
        print("\n" + "=" * 80)
        print(f"Finished test: {self._testMethodName}")
        print("=" * 80)
    
    def test_arithmetic_operations(self):
        """Test arithmetic operations on tensors."""
        # Create tensors on different devices
        cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
        gpu_tensor = torch.tensor([4.0, 5.0, 6.0], device='cuda')
        
        # Move CPU tensor to same device as GPU tensor for operations
        cpu_tensor_on_gpu = cpu_tensor.to(gpu_tensor.device)
        
        # Perform arithmetic operations
        result1 = cpu_tensor_on_gpu + gpu_tensor  # Addition between tensors on same device
        result2 = gpu_tensor * 2.0  # Multiplication with scalar
        result3 = torch.matmul(result1.view(1, 3), result2.view(3, 1))  # Matrix multiplication
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Calculate and print result for verification
        result_value = result3.item()
        print(f"Matrix multiplication result: {result_value}")
        
        # Get the captured log output
        self.log_capture.log_stream.getvalue()
        
        # Perform the diff check
        diff_check(self.log_capture)
    
    def test_tensor_reshaping(self):
        """Test tensor reshaping operations."""
        # Create a tensor on GPU
        tensor = torch.randn(4, 4, device='cuda')
        
        # Perform reshaping operations
        reshaped = tensor.view(2, 8)
        transposed = tensor.transpose(0, 1)
        permuted = tensor.permute(1, 0)
        
        # Get chunks and stack them
        chunks = torch.chunk(tensor, 2, dim=0)
        stacked = torch.stack(chunks, dim=0)
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Print shapes for verification
        print(f"Original shape: {tensor.shape}")
        print(f"Reshaped shape: {reshaped.shape}")
        print(f"Transposed shape: {transposed.shape}")
        print(f"Permuted shape: {permuted.shape}")
        print(f"Stacked shape: {stacked.shape}")
        
        # Get the captured log output
        self.log_capture.log_stream.getvalue()
        
        # Perform the diff check
        diff_check(self.log_capture)
    
    def test_tensor_indexing(self):
        """Test tensor indexing and slicing operations."""
        # Create a tensor on GPU
        tensor = torch.randn(4, 6, device='cuda')
        
        # Perform indexing operations
        slice1 = tensor[0:2, 1:3]
        slice2 = tensor[:, ::2]
        
        # Advanced indexing
        indices = torch.tensor([0, 2], device='cuda')
        indexed = tensor[indices]
        
        # Masked indexing
        mask = tensor > 0
        masked = tensor[mask]
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Print some values for verification
        print(f"Original tensor shape: {tensor.shape}")
        print(f"Slice1 shape: {slice1.shape}")
        print(f"Slice2 shape: {slice2.shape}")
        print(f"Indexed shape: {indexed.shape}")
        print(f"Masked tensor size: {masked.size(0)}")
        
        # Get the captured log output
        self.log_capture.log_stream.getvalue()
        
        # Perform the diff check
        diff_check(self.log_capture)


if __name__ == "__main__":
    unittest.main() 