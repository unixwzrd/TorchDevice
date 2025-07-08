"""
Common test utilities for TorchDevice tests.

This module provides the PrefixedTestCase base class for all tests,
which includes the 'info' method for logging test information.
"""
import unittest
import torch
import numpy as np
import random
import logging
import os
import sys
from pathlib import Path
from .log_diff import setup_log_capture, teardown_log_capture, diff_check

# --- Test-only state management ---

# Import internal components for test-specific setup
from TorchDevice.core.logger import set_redirect_log_stream


__all__ = ['PrefixedTestCase', 'diff_check', 'devices_equivalent', 'set_deterministic_seed']


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
    _test_dir = None

    @classmethod
    def setUpClass(cls):
        # Determine the test directory once per class
        cls._test_dir = Path(__file__).parent.parent.resolve()

    def setUp(self):
        super().setUp()
        # Set up a log capture stream for each test
        self.log_capture = setup_log_capture(self._testMethodName, self._test_dir)
        self.logger = logging.getLogger(self._testMethodName)

        # Set the redirect logger to output to our captured log file
        set_redirect_log_stream(self.log_capture.log_stream)

        # Set deterministic seeds
        set_deterministic_seed()

        # Print test header
        print("\n" + "=" * 80)
        self.logger.info(f"Starting test: {self.id()}")
        print("-" * 80)

    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        # Print a footer for the test
        print("-" * 80)
        self.logger.info(f"Finished test: {self.id()}")
        print("=" * 80)

        if hasattr(self, 'log_capture'):
            # Remove the test-specific log handler from the redirect logger.
            # This ensures no more logs are written to the stream before diffing.
            set_redirect_log_stream(None)

            # First, perform the diff check or update the expected log file,
            # but only if the expected file exists or we are in update mode.
            if self.log_capture.expected_output_file.exists() or os.environ.get('TORCHDEVICE_UPDATE_EXPECTED') == '1':
                diff_check(self.log_capture)

            # Then, tear down the capture and restore the original logger.
            teardown_log_capture(self.log_capture)

        super().tearDown()

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