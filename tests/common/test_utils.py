"""
Common test utilities for TorchDevice tests.

This module provides the PrefixedTestCase base class for all tests,
which includes the 'info' method for logging test information.
"""
import unittest
from pathlib import Path
from common.log_diff import setup_log_capture, teardown_log_capture


class PrefixedTestCase(unittest.TestCase):
    """
    Base test case that provides logging capabilities with test name prefixes.
    Also handles the log capture setup and teardown for test cases.
    """

    def setUp(self):
        """Set up test environment and logger capture."""
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