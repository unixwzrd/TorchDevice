"""
Test script to verify TDLogger functionality in isolation.
"""

import os
import sys
import unittest
from typing import Optional

# Add the parent directory to the Python path so we can import TorchDevice
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchDevice.tdlogger import TDLogger


class TestTDLogger(unittest.TestCase):
    """Test cases for TDLogger functionality."""

    def setUp(self):
        """Set up test environment."""
        self.logger = TDLogger()
        self.logger.enable()  # Ensure logging is enabled

    def test_basic_logging(self):
        """Test basic logging functionality."""
        self.logger.log_message("Test message")
        # Note: We can't easily capture stdout in unittest, so we're mainly testing that it doesn't raise exceptions

    def test_logger_with_intercepted_function(self):
        """Test logger with an intercepted function."""
        def test_function():
            pass
        self.logger(test_function)
        self.logger.log_message("Message with intercepted function")
        self.assertEqual(self.logger._intercepted_function, "test_function")

    def test_logger_disable_enable(self):
        """Test logger enable/disable functionality."""
        self.logger.disable()
        self.logger.log_message("This should not appear")
        self.logger.enable()
        self.logger.log_message("This should appear")

    def test_logger_with_submodule(self):
        """Test logger with a submodule structure."""
        class SubModule:
            def __init__(self):
                self.logger = TDLogger()
                self.logger(self.test_method)

            def test_method(self):
                pass

            def log_something(self):
                self.logger.log_message("Message from submodule")

        submodule = SubModule()
        submodule.log_something()
        self.assertEqual(submodule.logger._intercepted_function, "test_method")

    def test_logger_with_nested_calls(self):
        """Test logger with nested function calls."""
        def outer_function():
            self.logger(outer_function)
            self.logger.log_message("Outer message")
            
            def inner_function():
                self.logger(inner_function)
                self.logger.log_message("Inner message")
            
            inner_function()

        outer_function()
        # The last intercepted function should be inner_function
        self.assertEqual(self.logger._intercepted_function, "inner_function")

    def test_logger_with_multiple_instances(self):
        """Test multiple logger instances."""
        logger1 = TDLogger()
        logger2 = TDLogger()
        
        def test_function():
            pass
        
        logger1(test_function)
        logger2.log_message("Message from logger2")
        
        self.assertEqual(logger1._intercepted_function, "test_function")
        self.assertIsNone(logger2._intercepted_function)

    def test_logger_with_caller_info(self):
        """Test logger with caller information."""
        def test_function():
            self.logger.log_message("Message with caller info")
        
        test_function()
        # The message should include caller information
        # Note: We can't easily verify the exact caller info in the test

    def test_logger_with_empty_message(self):
        """Test logger with empty message."""
        self.logger.log_message("")
        # Should not raise any exceptions

    def test_logger_with_none_intercepted_function(self):
        """Test logger with None as intercepted function."""
        self.logger(None)
        self.logger.log_message("Message with None function")
        self.assertIsNone(self.logger._intercepted_function)


def main():
    """Run the tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    main() 