#!/usr/bin/env python
"""
Utility classes and functions for TorchDevice tests.
"""

import unittest
import logging
import sys


class PrefixedTestCase(unittest.TestCase):
    """
    A TestCase subclass that adds a prefix to test method names in logs.
    This helps identify which test is running in log output.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize console handler
        self.console_handler = None
    
    def info(self, msg, *args, **kwargs):
        """Log an info message with the test method name as a prefix."""
        self.logger.info(f"[{self._testMethodName}] {msg}", *args, **kwargs)
    
    def log_debug(self, msg, *args, **kwargs):
        """Log a debug message with the test method name as a prefix."""
        self.logger.debug(f"[{self._testMethodName}] {msg}", *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """Log a warning message with the test method name as a prefix."""
        self.logger.warning(f"[{self._testMethodName}] {msg}", *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """Log an error message with the test method name as a prefix."""
        self.logger.error(f"[{self._testMethodName}] {msg}", *args, **kwargs)
    
    def setUp(self):
        """Set up test case with prefixed logging."""
        # Add a console handler to display logs in the console
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(logging.INFO)
        
        # Get the logger
        self.logger = logging.getLogger("TorchDevice")
        
        # Save the original handlers
        self.original_handlers = self.logger.handlers.copy()
        self.original_level = self.logger.level
        
        # Add our console handler
        self.logger.addHandler(self.console_handler)
        self.logger.setLevel(logging.INFO)
        
        # Print a header for the test
        print(f"\n{'='*80}\nRunning test: {self._testMethodName}\n{'='*80}")
    
    def tearDown(self):
        """Clean up after the test."""
        # Remove our handler
        if hasattr(self, 'console_handler') and self.console_handler and hasattr(self, 'logger'):
            if self.console_handler in self.logger.handlers:
                self.logger.removeHandler(self.console_handler)
            
            # Restore the original handlers and level
            if hasattr(self, 'original_handlers') and hasattr(self, 'original_level'):
                self.logger.handlers = self.original_handlers
                self.logger.setLevel(self.original_level)
        
        # Print a footer for the test
        print(f"\n{'='*80}\nFinished test: {self._testMethodName}\n{'='*80}") 