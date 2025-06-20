"""
Test CUDA device utilities and properties with TorchDevice.

This module tests the logging of various CUDA utility functions with TorchDevice,
including device properties, device management, and memory operations.
"""
import logging
import unittest
import sys
from pathlib import Path

import torch
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture

import TorchDevice  # Import TorchDevice to ensure CUDA redirection is set up


class TestDeviceUtils(unittest.TestCase):
    """Test CUDA device utilities and properties with TorchDevice."""
    
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
    
    def test_device_properties(self):
        """Test device property functions."""
        # Test availability, device count, and current device
        is_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        
        # Test device properties
        props = torch.cuda.get_device_properties(0)
        
        # Test architecture and device capability
        try:
            arch_list = torch.cuda.get_arch_list()
        except AttributeError:
            # Handle the case where get_arch_list is not available
            arch_list = []
        
        try:
            device_cap = torch.cuda.get_device_capability(0)
        except AttributeError:
            # Handle the case where get_device_capability is not available
            device_cap = (0, 0)
        
        # Test device name
        device_name = torch.cuda.get_device_name(0)
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Print the results for verification
        print(f"CUDA Available: {is_available}")
        print(f"Device Count: {device_count}")
        print(f"Current Device: {current_device}")
        print(f"Device Properties: {props}")
        print(f"Architecture List: {arch_list}")
        print(f"Device Capability: {device_cap}")
        print(f"Device Name: {device_name}")
        
        # Get the captured log output
        self.log_capture.log_stream.getvalue()        

        # Perform the diff check
        diff_check(self.log_capture)
    
    def test_memory_functions(self):
        """Test memory-related functions."""
        # Create some tensors to allocate memory
        _ = torch.randn(1000, 1000, device='cuda')
        
        # Test memory functions
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            max_reserved = torch.cuda.max_memory_reserved()
        except (AttributeError, RuntimeError):
            # Handle the case where memory functions are not available
            allocated = 0
            reserved = 0
            max_allocated = 0
            max_reserved = 0
        
        # Test memory stats and snapshot
        try:
            _ = torch.cuda.memory_stats()
        except (AttributeError, RuntimeError):
            # Handle the case where memory_stats is not available
            pass
        
        try:
            _ = torch.cuda.memory_snapshot()
        except (AttributeError, RuntimeError):
            # Handle the case where memory_snapshot is not available
            pass
        
        # Test memory summary
        try:
            _ = torch.cuda.memory_summary()
        except (AttributeError, RuntimeError):
            # Handle the case where memory_summary is not available
            pass
        
        # Test empty cache
        torch.cuda.empty_cache()
        
        # Test reset peak memory stats
        try:
            torch.cuda.reset_peak_memory_stats()
        except (AttributeError, RuntimeError):
            # Handle the case where reset_peak_memory_stats is not available
            pass
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Print the results for verification
        print(f"Memory Allocated: {allocated}")
        print(f"Memory Reserved: {reserved}")
        print(f"Max Memory Allocated: {max_allocated}")
        print(f"Max Memory Reserved: {max_reserved}")
        
        # Get the captured log output
        self.log_capture.log_stream.getvalue()        
        
        # Perform the diff check
        diff_check(self.log_capture)
    
    def test_stream_and_events(self):
        """Test CUDA stream and event operations."""
        # Create a stream and events
        stream = torch.cuda.Stream()
        event1 = torch.cuda.Event(enable_timing=True)
        event2 = torch.cuda.Event(enable_timing=True)
        
        # Record event before operations
        event1.record(stream)
        
        # Perform operations on the stream
        with torch.cuda.stream(stream):
            tensor = torch.randn(1000, 1000, device='cuda')
            _ = tensor * 2
        
        # Record event after operations
        event2.record(stream)
        
        # Synchronize events
        event1.synchronize()
        event2.synchronize()
        
        # Calculate elapsed time
        elapsed_time = event1.elapsed_time(event2)
        
        # Test stream synchronize
        stream.synchronize()
        
        # Test stream wait_event
        stream.wait_event(event2)
        
        # Create another stream
        stream2 = torch.cuda.Stream()
        
        # Test stream wait_stream
        stream2.wait_stream(stream)
        
        # Test current and default stream
        _ = torch.cuda.current_stream()
        _ = torch.cuda.default_stream()
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Print the results for verification
        print(f"Elapsed Time: {elapsed_time} ms")
        
        # Get the captured log output
        self.log_capture.log_stream.getvalue()        
        
        # Perform the diff check
        diff_check(self.log_capture)


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])