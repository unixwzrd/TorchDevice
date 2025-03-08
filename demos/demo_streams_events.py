#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for TorchDevice stream and event functionality.

This script demonstrates how TorchDevice handles CUDA streams and events,
allowing them to work transparently on MPS devices.
"""

import os
import time
import TorchDevice  # Import TorchDevice first to apply patches
import torch

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def demo_basic_stream():
    """Demonstrate basic stream operations."""
    print_separator("Basic Stream Operations")
    
    # Create a CUDA stream (will be redirected to MPS if on Apple Silicon)
    stream = torch.cuda.Stream()
    print(f"Created stream: {stream}")
    
    # Check if stream is the current stream
    print(f"Is current stream: {stream.cuda_stream == torch.cuda.current_stream().cuda_stream}")
    
    # Use stream as context manager
    print("\nUsing stream as context manager:")
    with torch.cuda.stream(stream):
        print(f"Current stream inside context: {torch.cuda.current_stream()}")
        # Do some work in the stream
        x = torch.randn(1000, 1000, device='cuda')
        y = x @ x.T
    
    print(f"Current stream after context: {torch.cuda.current_stream()}")
    
    # Synchronize the stream
    print("\nSynchronizing stream...")
    stream.synchronize()
    print("Stream synchronized")

def demo_events():
    """Demonstrate event operations."""
    print_separator("Event Operations")
    
    # Create CUDA events (will be redirected to MPS if on Apple Silicon)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print(f"Created start event: {start_event}")
    print(f"Created end event: {end_event}")
    
    # Record events and measure time
    print("\nMeasuring execution time with events:")
    start_event.record()
    
    # Do some work
    matrix_size = 2000
    a = torch.randn(matrix_size, matrix_size, device='cuda')
    b = torch.randn(matrix_size, matrix_size, device='cuda')
    c = a @ b
    
    end_event.record()
    
    # Wait for events to complete
    print("Waiting for events to complete...")
    torch.cuda.synchronize()
    
    # Calculate elapsed time
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"Matrix multiplication of size {matrix_size}x{matrix_size} took {elapsed_time:.2f} ms")

def demo_multiple_streams():
    """Demonstrate using multiple streams for concurrent operations."""
    print_separator("Multiple Streams for Concurrent Operations")
    
    # Create multiple streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    print(f"Created stream1: {stream1}")
    print(f"Created stream2: {stream2}")
    
    # Record start event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event1 = torch.cuda.Event(enable_timing=True)
    end_event2 = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    # Execute operations in first stream
    with torch.cuda.stream(stream1):
        print("Executing operations in stream1...")
        a = torch.randn(2000, 2000, device='cuda')
        b = torch.randn(2000, 2000, device='cuda')
        c = a @ b
        end_event1.record()
    
    # Execute operations in second stream
    with torch.cuda.stream(stream2):
        print("Executing operations in stream2...")
        x = torch.randn(2000, 2000, device='cuda')
        y = torch.randn(2000, 2000, device='cuda')
        z = x @ y
        end_event2.record()
    
    # Wait for all operations to complete
    torch.cuda.synchronize()
    
    # Calculate elapsed times
    elapsed_time1 = start_event.elapsed_time(end_event1)
    elapsed_time2 = start_event.elapsed_time(end_event2)
    
    print(f"Stream1 operations took {elapsed_time1:.2f} ms")
    print(f"Stream2 operations took {elapsed_time2:.2f} ms")

def demo_stream_dependencies():
    """Demonstrate stream dependencies with wait_stream."""
    print_separator("Stream Dependencies with wait_stream")
    
    # Create streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    # Create events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    middle_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Start timing
    start_event.record()
    
    # Execute first operation in stream1
    with torch.cuda.stream(stream1):
        print("Stream1: Starting first operation...")
        a = torch.randn(2000, 2000, device='cuda')
        b = torch.randn(2000, 2000, device='cuda')
        c = a @ b
        print("Stream1: First operation completed")
    
    # Record middle event
    middle_event.record(stream1)
    
    # Make stream2 wait for stream1 to complete
    stream2.wait_stream(stream1)
    
    # Execute second operation in stream2 (will wait for stream1)
    with torch.cuda.stream(stream2):
        print("Stream2: Waiting for stream1 to complete...")
        print("Stream2: Starting second operation...")
        x = torch.randn(2000, 2000, device='cuda')
        y = torch.randn(2000, 2000, device='cuda')
        z = x @ y
        print("Stream2: Second operation completed")
    
    # Record end event
    end_event.record(stream2)
    
    # Wait for all operations to complete
    torch.cuda.synchronize()
    
    # Calculate elapsed times
    first_op_time = start_event.elapsed_time(middle_event)
    second_op_time = middle_event.elapsed_time(end_event)
    total_time = start_event.elapsed_time(end_event)
    
    print(f"First operation took {first_op_time:.2f} ms")
    print(f"Second operation took {second_op_time:.2f} ms")
    print(f"Total execution time: {total_time:.2f} ms")

def main():
    """Run all demos."""
    # Print device information
    device = torch.device('cuda')  # Will be redirected to MPS on Apple Silicon if CUDA is not available
    print(f"Using device: {device}")
    
    # Run demos
    demo_basic_stream()
    demo_events()
    demo_multiple_streams()
    demo_stream_dependencies()

if __name__ == "__main__":
    main()
