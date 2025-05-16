"""
Common test utilities for TorchDevice tests.

This module provides the PrefixedTestCase base class for all tests,
which includes the 'info' method for logging test information.
"""
import unittest
import torch
import numpy as np
import random
from pathlib import Path
from common.log_diff import setup_log_capture, teardown_log_capture


def set_deterministic_seed(seed=42):
    """
    Set deterministic seeds for reproducible tests.
    
    This sets seeds for Python's random module, NumPy, PyTorch CPU, 
    and PyTorch CUDA/MPS if available.
    
    Args:
        seed: The seed value to use (default: 42)
    """
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed (CPU and all devices)
    # Use standard PyTorch interfaces
    torch.manual_seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set MPS seed if available
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        if hasattr(torch.mps, 'manual_seed'):
            torch.mps.manual_seed(seed)
    
    # Extra deterministic settings for PyTorch
    if hasattr(torch.backends.cudnn, 'deterministic'):
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cudnn, 'benchmark'):
        torch.backends.cudnn.benchmark = False
    
    print(f"Set deterministic seed: {seed} for random, numpy, and torch")


class PrefixedTestCase(unittest.TestCase):
    """
    Base test case that provides logging capabilities with test name prefixes.
    Also handles the log capture setup and teardown for test cases.
    """

    def setUp(self):
        """Set up test environment and logger capture."""
        # Set deterministic seeds
        set_deterministic_seed()
        
        # Set up log capture for TDLogger
        test_dir = Path(__file__).parent.parent
        self.log_capture = setup_log_capture(self._testMethodName, test_dir)
        
        # Print test header
        print("\n" + "=" * 80)
        print(f"Starting test: {self._testMethodName}")
        print("=" * 80)

    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        if hasattr(self, 'log_capture'):
            teardown_log_capture(self.log_capture)
        
        # Print a footer for the test
        print("\n" + "=" * 80)
        print(f"Finished test: {self._testMethodName}")
        print("=" * 80)

    def info(self, msg, *args, **kwargs):
        """
        Print an informational message for this test.
        
        Args:
            msg: The message format string
            *args: Arguments to be formatted into the message
            **kwargs: Keyword arguments passed to print
        """
        if args:
            msg = msg % args
        print(f"TEST INFO - {self.__class__.__name__}.{self._testMethodName}: {msg}", **kwargs)

    def print_debug(self, msg, *args, **kwargs):
        """
        Print a debug message to stdout directly.
        Only use this for test diagnostics that should not be captured in log files.
        
        Args:
            msg: The message format string
            *args: Arguments to be formatted into the message
            **kwargs: Keyword arguments passed to print
        """
        if args:
            msg = msg % args
        print(f"TEST DEBUG - {self.__class__.__name__}.{self._testMethodName}: {msg}", **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Print a warning message for this test.
        
        Args:
            msg: The message format string
            *args: Arguments to be formatted into the message
            **kwargs: Keyword arguments passed to print
        """
        if args:
            msg = msg % args
        print(f"TEST WARNING - {self.__class__.__name__}.{self._testMethodName}: {msg}", **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Print an error message for this test.
        
        Args:
            msg: The message format string
            *args: Arguments to be formatted into the message
            **kwargs: Keyword arguments passed to print
        """
        if args:
            msg = msg % args
        print(f"TEST ERROR - {self.__class__.__name__}.{self._testMethodName}: {msg}", **kwargs)

def devices_equivalent(a, b):
    """
    Robust device comparison: for CPU and MPS, index 0 and None are equivalent;
    for CUDA, index must match.
    Accepts torch.device or string.
    """
    import torch
    if isinstance(a, str):
        a = torch.device(a)
    if isinstance(b, str):
        b = torch.device(b)
    if a.type != b.type:
        return False
    if a.type in ("cpu", "mps"):
        return (a.index in (None, 0)) and (b.index in (None, 0))
    return a.index == b.index 