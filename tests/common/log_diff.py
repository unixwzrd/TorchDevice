"""
Utility functions for TDLogger tests.

This module provides common functionality for testing the TDLogger module,
including log capture and comparison utilities.
"""
from dataclasses import dataclass
import argparse
import difflib
import io
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

UPDATE_EXPECTED_OUTPUT = False


def check_args():
    """
    Check command line arguments or environment variables for the update-expected flag.
    """
    global UPDATE_EXPECTED_OUTPUT
    # Check environment variable first, as it's more reliable for subprocesses.
    if os.environ.get('TORCHDEVICE_UPDATE_EXPECTED') == '1':
        UPDATE_EXPECTED_OUTPUT = True
        return

    # Fallback to command-line arguments for standalone test runs.
    parser = argparse.ArgumentParser(description='Run TDLogger tests')
    parser.add_argument('--update-expected', action='store_true', help='Update expected output files', default=False)

    # Parse known args to avoid conflicts with unittest's own argument parsing.
    args, remaining = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining  # Remove our custom args so unittest doesn't see them.

    # Global flag to update expected outputs.
    UPDATE_EXPECTED_OUTPUT = args.update_expected


@dataclass
class LogCapture:
    logger: logging.Logger
    log_stream: io.StringIO
    log_handler: logging.Handler
    console_handler: logging.Handler
    original_handlers: list
    original_level: int
    expected_output_file: Path
    temp_output_file: Path


def diff_check(log_capture: LogCapture) -> None:
    """
    Compare the captured log in log_capture.log_stream with the expected output file.
    If UPDATE_EXPECTED_OUTPUT is True, update the expected file.
    Otherwise, raise an AssertionError if there’s any diff.
    
    Args:
        log_capture (LogCapture): The LogCapture object containing logger state and file paths.
    """
    # Get the captured log output from the StringIO stream.
    captured_log = log_capture.log_stream.getvalue()
    expected_file: Path = log_capture.expected_output_file

    # Create a temporary file path for diff comparison.
    temp_file: Path = log_capture.temp_output_file.with_name(
        log_capture.temp_output_file.stem + "_temp.log"
    )
    
    # Write the captured log to the temporary file.
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(captured_log)
    
    # Use the global UPDATE_EXPECTED_OUTPUT flag.
    global UPDATE_EXPECTED_OUTPUT
    if UPDATE_EXPECTED_OUTPUT:
        expected_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(temp_file, expected_file)
        print(f"Updated expected output file: {expected_file}")
        return

    if not expected_file.exists():
        raise AssertionError(f"Expected output file does not exist: {expected_file}")

    with open(expected_file, 'r', encoding='utf-8') as f:
        expected_lines = f.readlines()
    with open(temp_file, 'r', encoding='utf-8') as f:
        temp_lines = f.readlines()

    diff = ''.join(difflib.unified_diff(
        expected_lines,
        temp_lines,
        fromfile=str(expected_file),
        tofile=str(temp_file)
    ))
    
    if diff:
        raise AssertionError(f"Log output differs from expected output:\n{diff}")


def setup_log_capture(test_name: str, base_dir: Path) -> LogCapture:
    """
    Set up log capture for TDLogger tests.

    Args:
        test_name: Unique name for the test (e.g., self._testMethodName)
        base_dir: Directory where the expected and temporary log files should be placed

    Returns:
        LogCapture: An object containing the logger, log stream, handlers, and file paths.
    """
    logger = logging.getLogger("TorchDevice")
    original_handlers = logger.handlers.copy()
    original_level = logger.level

    # Remove all existing handlers to prevent duplication.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(logging.DEBUG)

    # Create a StringIO object to capture log output.
    log_stream = io.StringIO()

    # Define a formatter that matches your TDLogger format.
    formatter = logging.Formatter(
        'GPU REDIRECT - [%(program_name)s] "%(caller_func_name)s" in File: %(caller_filename)s:%(caller_lineno)s - '
        'Called: %(torch_function)s %(message)s'
    )

    # Create a handler that writes to the StringIO object.
    log_handler = logging.StreamHandler(log_stream)
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(formatter)

    # Create a handler that writes to the console.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger.
    logger.addHandler(log_handler)
    logger.addHandler(console_handler)

    # Define expected and temporary output file paths.
    expected_output_file = base_dir / f"{test_name}_expected.log"
    temp_output_file = base_dir / f"{test_name}_temp.log"
    if temp_output_file.exists():
        temp_output_file.unlink()

    return LogCapture(
        logger=logger,
        log_stream=log_stream,
        log_handler=log_handler,
        console_handler=console_handler,
        original_handlers=original_handlers,
        original_level=original_level,
        expected_output_file=expected_output_file,
        temp_output_file=temp_output_file
    )

def teardown_log_capture(log_capture: LogCapture) -> None:
    """
    Restore the original logger configuration.
    
    Args:
        log_capture (LogCapture): The object returned by setup_log_capture.
    """
    logger = log_capture.logger
    # Flush our handlers.
    log_capture.log_handler.flush()
    # Remove the added handlers.
    for handler in (log_capture.log_handler, log_capture.console_handler):
        if handler in logger.handlers:
            logger.removeHandler(handler)
    # Restore original configuration.
    logger.handlers = log_capture.original_handlers
    logger.setLevel(log_capture.original_level)


# Let's check the arguments and pull our --update-expected flag and set the global
# variable UPDATE_EXPECTED_OUTPUT if it is, and remove it from the sys.argv list.
check_args()
