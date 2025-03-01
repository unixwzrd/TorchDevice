#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import logging
import functools
import inspect

# Get the logger
logger = logging.getLogger(__name__)

class PrefixedTestCase(unittest.TestCase):
    """
    A custom TestCase that adds the test name as a prefix to log messages.
    This makes it easier to identify which test is generating which log messages.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_logger = logging.getLogger(self.__class__.__name__)
    
    def info(self, msg, *args, **kwargs):
        """Log an info message with the test prefix."""
        prefix = f"[TEST {self.__class__.__name__}.{self._testMethodName}]"
        self.test_logger.info(f"{prefix} {msg}", *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        """Log a debug message with the test prefix."""
        prefix = f"[TEST {self.__class__.__name__}.{self._testMethodName}]"
        self.test_logger.debug(f"{prefix} {msg}", *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """Log a warning message with the test prefix."""
        prefix = f"[TEST {self.__class__.__name__}.{self._testMethodName}]"
        self.test_logger.warning(f"{prefix} {msg}", *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """Log an error message with the test prefix."""
        prefix = f"[TEST {self.__class__.__name__}.{self._testMethodName}]"
        self.test_logger.error(f"{prefix} {msg}", *args, **kwargs)
