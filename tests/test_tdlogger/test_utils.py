"""
Utility functions for TDLogger tests.

This module provides common functionality for testing the TDLogger module,
including log capture and comparison utilities.
"""

import argparse
import difflib
import io
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run TDLogger tests')
parser.add_argument('--update-expected', action='store_true', help='Update expected output files')

# Parse known args to avoid conflicts with unittest's own argument parsing
args, remaining = parser.parse_known_args()
sys.argv = sys.argv[:1] + remaining  # Remove our custom args so unittest doesn't see them

# Global flag to update expected outputs.
UPDATE_EXPECTED_OUTPUT = args.update_expected or os.environ.get('UPDATE_EXPECTED_OUTPUT') == '1'


def diff_check(captured_log: str, expected_file: Path) -> None:
    """
    If UPDATE_EXPECTED_OUTPUT is True, update the expected file with the captured log.
    Otherwise, compare the captured log with the expected file and raise an AssertionError if they differ.
    
    Args:
        captured_log: The log output captured during the test.
        expected_file: Path to the expected log file.
    """
    # Create the temp file path
    temp_file = expected_file.with_name(expected_file.stem + "_temp.log")
    
    # Write the captured log to the temp file
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(captured_log)
    
    # If we're updating the expected output, copy the temp file to the expected file
    if UPDATE_EXPECTED_OUTPUT:
        # Make sure the parent directory exists
        expected_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the temp file to the expected file
        shutil.copy(temp_file, expected_file)
        print(f"Updated expected output file: {expected_file}")
        return
    
    # If the expected file doesn't exist, fail the test
    if not expected_file.exists():
        raise AssertionError(f"Expected output file does not exist: {expected_file}")
    
    # Compare the temp file with the expected file
    with open(expected_file, 'r', encoding='utf-8') as expected_f:
        expected_lines = expected_f.readlines()
    
    with open(temp_file, 'r', encoding='utf-8') as temp_f:
        temp_lines = temp_f.readlines()
    
    # Use difflib to get a nice diff
    diff = difflib.unified_diff(
        expected_lines, 
        temp_lines,
        fromfile=str(expected_file),
        tofile=str(temp_file)
    )
    
    # Convert the diff to a string
    diff_text = ''.join(list(diff))
    
    # If there are differences, fail the test
    if diff_text:
        raise AssertionError(f"Log output differs from expected output:\n{diff_text}")


def setup_log_capture() -> Tuple[logging.Logger, io.StringIO, logging.Handler, logging.Handler, list, int]:
    """
    Set up log capture for TDLogger tests.
    
    Returns:
        Tuple containing:
        - logger: The configured logger
        - log_stream: StringIO object capturing log output
        - log_handler: Handler for capturing logs
        - console_handler: Handler for displaying logs in console
        - original_handlers: The original handlers to restore in teardown
        - original_level: The original level to restore in teardown
    """
    # Get the logger
    logger = logging.getLogger("TorchDevice")
    
    # Save the original handlers and level
    original_handlers = logger.handlers.copy()
    original_level = logger.level
    
    # Set the logger level to DEBUG
    logger.setLevel(logging.DEBUG)
    
    # Create a StringIO object to capture log output
    log_stream = io.StringIO()
    
    # Create a handler that writes to the StringIO object
    log_handler = logging.StreamHandler(log_stream)
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Create a handler that writes to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Add the handlers to the logger
    logger.addHandler(log_handler)
    logger.addHandler(console_handler)
    
    return logger, log_stream, log_handler, console_handler, original_handlers, original_level


def teardown_log_capture(
    logger: logging.Logger, 
    original_handlers: list, 
    original_level: int,
    handlers_to_remove: list
) -> None:
    """
    Clean up after log capture.
    
    Args:
        logger: The logger to clean up
        original_handlers: The original handlers to restore
        original_level: The original level to restore
        handlers_to_remove: Handlers to remove from the logger
    """
    # Remove our handlers
    for handler in handlers_to_remove:
        if handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Restore the original handlers and level
    logger.handlers = original_handlers
    logger.setLevel(original_level) 