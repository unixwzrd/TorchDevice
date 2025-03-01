#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import torch
import TorchDevice
import logging
import time
import sys
import os

# Add the current directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_utils import PrefixedTestCase  # Import our custom TestCase

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Ensure this configuration is applied
)
logger = logging.getLogger(__name__)

# Prevent duplicate logging
logging.getLogger("TorchDevice").propagate = False

class TestCUDAOperations(PrefixedTestCase):
    """
    Test class for CUDA operations with TorchDevice.
    This combines tests from the test_projects directory to ensure
    comprehensive testing of CUDA functionality in the main test suite.
    """

    def setUp(self):
        """Set up test environment."""
        # Ensure TorchDevice is initialized
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.info("Using device: %s", self.device)
        
    def test_tensor_operations(self):
        """Test basic tensor operations on the device."""
        self.info("Starting tensor operations test")
        
        # Create a tensor on the device
        x = torch.randn(100, device=self.device)
        self.info("Created random tensor with shape %s on %s", x.shape, self.device)
        
        # Test square operation
        y = x * x
        self.info("Performed square operation")
        
        # Compute reference result on CPU
        x_cpu = x.cpu()
        y_ref = x_cpu * x_cpu
        
        # Verify results
        self.assertTrue(torch.allclose(y.cpu(), y_ref))
        self.info("Verified tensor operations results")

    def test_matrix_multiplication(self):
        """Test matrix multiplication on the device."""
        # Create tensors on the device
        a = torch.randn(50, 50, device=self.device)
        b = torch.randn(50, 50, device=self.device)
        
        # Perform matrix multiplication
        c = torch.matmul(a, b)
        
        # Compute reference result on CPU
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        
        # Compare results
        self.assertTrue(torch.allclose(c.cpu(), c_cpu, rtol=1e-5, atol=1e-5))
        self.assertEqual(c.device, self.device)
        
    def test_cuda_stream_operations(self):
        """Test CUDA stream operations."""
        # Create a CUDA stream
        stream = torch.cuda.Stream()
        self.assertIsNotNone(stream)
        
        # Create a tensor on the device
        x = torch.randn(100, device=self.device)
        
        # Perform operation in stream
        with torch.cuda.stream(stream):
            y = x * 2
        
        # Synchronize stream
        stream.synchronize()
        
        # Query stream
        query_result = stream.query()
        self.assertTrue(query_result)
        
        # Get current stream
        current_stream = torch.cuda.current_stream()
        self.assertIsNotNone(current_stream)
        
        # Get default stream
        default_stream = torch.cuda.default_stream()
        self.assertIsNotNone(default_stream)
        
    def test_cuda_event_operations(self):
        """Test CUDA event operations."""
        self.info("Starting CUDA event operations test")
        
        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        self.info("Created CUDA events with timing enabled")
        
        # Create a tensor and perform operations
        tensor = torch.randn(1000, 1000, device=self.device)
        
        # Record start event
        start_event.record()
        self.info("Recorded start event")
        
        # Perform some operations
        result = tensor @ tensor
        self.info("Performed matrix multiplication")
        
        # Record end event
        end_event.record()
        self.info("Recorded end event")
        
        # Synchronize to ensure operations are complete
        torch.cuda.synchronize()
        self.info("Synchronized CUDA")
        
        # Get elapsed time
        elapsed_time = start_event.elapsed_time(end_event)
        self.info("Elapsed time: %s ms", elapsed_time)
        
        # Verify elapsed time is reasonable
        self.assertGreaterEqual(elapsed_time, 0.0)
        self.info("CUDA event operations test completed successfully")

    def test_cuda_stream_with_events(self):
        """Test CUDA streams with events."""
        # Create a CUDA stream
        stream = torch.cuda.Stream()
        
        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Create a tensor on the device
        x = torch.randn(1000, 1000, device=self.device)
        
        # Record start event with stream
        start_event.record(stream)
        
        # Perform operation in stream
        with torch.cuda.stream(stream):
            y = torch.matmul(x, x)
        
        # Record end event with stream
        end_event.record(stream)
        
        # Synchronize stream
        stream.synchronize()
        
        # Get elapsed time
        elapsed_time = start_event.elapsed_time(end_event)
        self.assertGreaterEqual(elapsed_time, 0)
        
    def test_multiple_streams(self):
        """Test multiple CUDA streams."""
        # Create multiple CUDA streams
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        # Create tensors on the device
        x1 = torch.randn(100, device=self.device)
        x2 = torch.randn(100, device=self.device)
        
        # Perform operations in different streams
        with torch.cuda.stream(stream1):
            y1 = x1 * 2
            
        with torch.cuda.stream(stream2):
            y2 = x2 * 3
            
        # Synchronize all streams
        torch.cuda.synchronize()
        
        # Verify results
        self.assertTrue(torch.allclose(y1.cpu(), (x1 * 2).cpu()))
        self.assertTrue(torch.allclose(y2.cpu(), (x2 * 3).cpu()))
        
    def test_stream_wait_event(self):
        """Test stream waiting for an event."""
        # Create CUDA streams
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        # Create a CUDA event
        event = torch.cuda.Event()
        
        # Create tensors on the device
        x = torch.randn(100, device=self.device)
        
        # Perform operation in first stream and record event
        with torch.cuda.stream(stream1):
            y = x * 2
            event.record(stream1)
            
        # Make second stream wait for the event
        stream2.wait_event(event)
        
        # Perform operation in second stream that depends on first operation
        with torch.cuda.stream(stream2):
            z = y * 3
            
        # Synchronize second stream
        stream2.synchronize()
        
        # Verify results
        expected = x * 2 * 3
        self.assertTrue(torch.allclose(z.cpu(), expected.cpu()))
        
    def test_stream_wait_stream(self):
        """Test stream waiting for another stream."""
        # Skip this test if wait_stream is not available
        if not hasattr(torch.cuda.Stream, 'wait_stream'):
            self.skipTest("wait_stream not available in this PyTorch version")
            
        # Create CUDA streams
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        # Create tensors on the device
        x = torch.randn(100, device=self.device)
        
        # Perform operation in first stream
        with torch.cuda.stream(stream1):
            y = x * 2
            
        # Make second stream wait for the first stream
        stream2.wait_stream(stream1)
        
        # Perform operation in second stream that depends on first operation
        with torch.cuda.stream(stream2):
            z = y * 3
            
        # Synchronize second stream
        stream2.synchronize()
        
        # Verify results
        expected = x * 2 * 3
        self.assertTrue(torch.allclose(z.cpu(), expected.cpu()))

if __name__ == '__main__':
    unittest.main()
