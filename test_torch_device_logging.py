"""
Test script to verify TorchDevice logging functionality.
"""

import torch
from TorchDevice import TorchDevice

def test_device_detection():
    """Test device detection and logging"""
    print("\n=== Testing Device Detection ===")
    device = TorchDevice()
    print(f"Created device: {device}")

def test_tensor_operations():
    """Test tensor operations and logging"""
    print("\n=== Testing Tensor Operations ===")
    # Create a tensor
    x = torch.randn(3, 3)
    print(f"Created tensor: {x}")
    
    # Move to device
    x = x.cuda()
    print(f"Tensor after cuda(): {x}")
    
    # Use to() method
    x = x.to('mps')
    print(f"Tensor after to('mps'): {x}")

def test_module_operations():
    """Test module operations and logging"""
    print("\n=== Testing Module Operations ===")
    # Create a simple module
    model = torch.nn.Linear(10, 5)
    print(f"Created model: {model}")
    
    # Move to device
    model = model.cuda()
    print(f"Model after cuda(): {model}")
    
    # Use to() method
    model = model.to('mps')
    print(f"Model after to('mps'): {model}")

def test_cuda_functions():
    """Test CUDA function mocking and logging"""
    print("\n=== Testing CUDA Functions ===")
    # Test various CUDA functions
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

def test_stream_operations():
    """Test stream operations and logging"""
    print("\n=== Testing Stream Operations ===")
    # Create a stream
    stream = torch.cuda.Stream()
    print(f"Created stream: {stream}")
    
    # Use stream context
    with torch.cuda.stream(stream):
        print("Inside stream context")
        x = torch.randn(3, 3).cuda()
        print(f"Created tensor in stream: {x}")

def test_event_operations():
    """Test event operations and logging"""
    print("\n=== Testing Event Operations ===")
    # Create events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    print(f"Created events: {start_event}, {end_event}")
    
    # Record events
    start_event.record()
    print("Recorded start event")
    
    # Simulate some work
    x = torch.randn(1000, 1000).cuda()
    x = x @ x.T
    
    end_event.record()
    print("Recorded end event")
    
    # Synchronize and get elapsed time
    end_event.synchronize()
    print(f"Elapsed time: {start_event.elapsed_time(end_event):.2f} ms")

def main():
    """Run all tests"""
    print("Starting TorchDevice logging tests...")
    
    # Run each test
    test_device_detection()
    test_tensor_operations()
    test_module_operations()
    test_cuda_functions()
    test_stream_operations()
    test_event_operations()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 