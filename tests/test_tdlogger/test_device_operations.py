"""
Test device operations with TorchDevice.

This module tests various device operations with TorchDevice,
ensuring that the correct log messages are generated.
"""
import unittest
import sys
from pathlib import Path

import torch
import torch.nn as nn
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture
import TorchDevice

# Import TorchDevice to ensure CUDA redirection is set up
_ = TorchDevice


class TestDeviceOperations(unittest.TestCase):
    """Test device operations with TorchDevice."""
    
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

    def test_tensor_creation(self):
        """
        Test tensor creation operations by generating log output and performing a diff check.
        """
        # Create a tensor on the CPU
        _ = torch.tensor([1.0, 2.0, 3.0])
        
        # Create a tensor on the GPU (will be redirected to MPS on Mac)
        _ = torch.tensor([4.0, 5.0, 6.0], device='cuda')
        
        # Create a tensor with specific dtype
        _ = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32)
        
        # Create a tensor with both device and dtype
        _ = torch.tensor([10.0, 11.0, 12.0], device='cuda', dtype=torch.float32)
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Get the captured log output
        self.log_capture.log_stream.getvalue()
        
        # Perform the diff check
        diff_check(self.log_capture)

    def test_tensor_movement(self):
        """
        Test tensor movement operations by generating log output and performing a diff check.
        """
        # Create tensors on different devices
        x_cpu = torch.tensor([1.0, 2.0, 3.0])
        x_cuda = torch.tensor([4.0, 5.0, 6.0], device='cuda')
        
        # Test various movement operations
        _ = x_cpu.cuda()  # CPU to CUDA
        _ = x_cuda.cpu()  # CUDA to CPU
        _ = x_cpu.to('cuda')  # Using to() method
        _ = x_cuda.to('cpu')  # Using to() method with string
        _ = x_cpu.to(device='cuda')  # Using to() with kwargs
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Get the captured log output
        self.log_capture.log_stream.getvalue()
        
        # Perform the diff check
        diff_check(self.log_capture)

    def test_cuda_functions(self):
        """
        Test CUDA functions by generating log output and performing a diff check.
        """
        # Check CUDA availability and properties
        _ = torch.cuda.is_available()
        print(f"Device: {_}")
        _ = torch.cuda.device_count()
        print(f"Device count: {_}")
        _ = torch.cuda.current_device()
        print(f"Current device: {_}")
        _ = torch.cuda.get_device_name()
        print(f"Device name: {_}")
        _ = torch.cuda.get_device_properties(0)
        print(f"Device properties: {_}")
        
        # Memory management
        torch.cuda.empty_cache()
        torch.cuda.memory_allocated()
        torch.cuda.memory_reserved()
        
        # Device management
        torch.cuda.synchronize()
        
        # Get the captured log output
        self.log_capture.log_stream.getvalue()
        
        # Perform the diff check
        diff_check(self.log_capture)

    def test_module_device_operations(self):
        """
        Test neural network module device operations by generating log output and performing a diff check.
        """
        # Create a simple neural network
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Move model to different devices
        model.cuda()  # Move to CUDA
        model.cpu()   # Move back to CPU
        model.to('cuda')  # Using to() method
        model.to(device='cpu')  # Using to() with kwargs
        
        # Create input and test forward pass on different devices
        x_cpu = torch.randn(2, 10)
        x_cuda = x_cpu.cuda()
        
        # Run model on different devices
        _ = model(x_cuda)  # Forward pass on CUDA
        model.cpu()
        _ = model(x_cpu)   # Forward pass on CPU
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Perform the diff check
        diff_check(self.log_capture)

    def test_device_context(self):
        """
        Test device context management by generating log output and performing a diff check.
        """
        # Test device context management
        with torch.cuda.device(0):
            _ = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            _ = torch.cuda.current_device()
        
        # Test device context with multiple operations
        with torch.cuda.device(0):
            x = torch.tensor([4.0, 5.0, 6.0], device='cuda')
            y = torch.tensor([7.0, 8.0, 9.0], device='cuda')
            _ = x + y
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Perform the diff check
        diff_check(self.log_capture)


if __name__ == '__main__':
    unittest.main(argv=sys.argv[:1])
