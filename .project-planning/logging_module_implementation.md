# Logging Module Implementation Plan

## Current Implementation
The current logging implementation is spread throughout the TorchDevice.py file and includes:
- `get_caller_info()` function for retrieving caller information
- `log_message()` function for formatting and printing log messages
- `log_info()`, `log_warning()`, and `log_error()` functions for different log levels
- Global variables for log verbosity and log file

## New Implementation

### 1. Create logging.py

```python
import os
import sys
import inspect
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# Log levels
LOG_LEVEL_INFO = 0
LOG_LEVEL_WARNING = 1
LOG_LEVEL_ERROR = 2

# Default log verbosity (0 = minimal, 1 = verbose, 2 = debug)
LOG_VERBOSITY = int(os.environ.get('TORCHDEVICE_LOG_VERBOSITY', '0'))

# Log file path
LOG_FILE = os.environ.get('TORCHDEVICE_LOG_FILE', None)

def get_caller_info() -> Dict[str, Any]:
    """
    Get information about the caller of the function.
    
    Returns:
        Dict[str, Any]: A dictionary containing caller information.
    """
    # Implementation from current TorchDevice.py
    caller_info = {}
    try:
        frame = inspect.currentframe()
        if frame:
            # Go up 3 frames to get the caller of the logging function
            frame = frame.f_back.f_back.f_back
            if frame:
                caller_info['file'] = frame.f_code.co_filename
                caller_info['line'] = frame.f_lineno
                caller_info['function'] = frame.f_code.co_name
                # Get the module name
                module = inspect.getmodule(frame)
                if module:
                    caller_info['module'] = module.__name__
                else:
                    caller_info['module'] = "unknown"
    except Exception as e:
        print(f"Error getting caller info: {e}")
    
    return caller_info

def log_message(level: int, message: str, torch_function: Optional[str] = None) -> None:
    """
    Log a message with the specified level.
    
    Args:
        level (int): The log level (0=INFO, 1=WARNING, 2=ERROR).
        message (str): The message to log.
        torch_function (Optional[str], optional): The torch function being called. Defaults to None.
    """
    # Implementation from current TorchDevice.py
    level_str = ["INFO", "WARNING", "ERROR"][level]
    caller_info = get_caller_info()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Format the message
    if torch_function:
        formatted_message = f"[{timestamp}] {level_str} [{torch_function}] {message}"
    else:
        formatted_message = f"[{timestamp}] {level_str} {message}"
    
    # Add caller info if available
    if caller_info:
        file = caller_info.get('file', 'unknown')
        line = caller_info.get('line', 'unknown')
        function = caller_info.get('function', 'unknown')
        formatted_message += f" (called from {file}:{line} in {function})"
    
    # Print to console
    print(formatted_message, file=sys.stderr)
    
    # Write to log file if specified
    if LOG_FILE:
        try:
            with open(LOG_FILE, 'a') as f:
                f.write(formatted_message + '\n')
        except Exception as e:
            print(f"Error writing to log file: {e}", file=sys.stderr)

def log_info(message: str, torch_function: Optional[str] = None) -> None:
    """
    Log an info message.
    
    Args:
        message (str): The message to log.
        torch_function (Optional[str], optional): The torch function being called. Defaults to None.
    """
    log_message(LOG_LEVEL_INFO, message, torch_function)

def log_warning(message: str, torch_function: Optional[str] = None) -> None:
    """
    Log a warning message.
    
    Args:
        message (str): The message to log.
        torch_function (Optional[str], optional): The torch function being called. Defaults to None.
    """
    log_message(LOG_LEVEL_WARNING, message, torch_function)

def log_error(message: str, torch_function: Optional[str] = None) -> None:
    """
    Log an error message.
    
    Args:
        message (str): The message to log.
        torch_function (Optional[str], optional): The torch function being called. Defaults to None.
    """
    log_message(LOG_LEVEL_ERROR, message, torch_function)

def set_verbosity(level: int) -> None:
    """
    Set the log verbosity level.
    
    Args:
        level (int): The verbosity level (0=minimal, 1=verbose, 2=debug).
    """
    global LOG_VERBOSITY
    LOG_VERBOSITY = level
    log_info(f"Log verbosity set to {level}", "TorchDevice.set_verbosity")

def set_log_file(file_path: Optional[str]) -> None:
    """
    Set the log file path.
    
    Args:
        file_path (Optional[str]): The path to the log file, or None to disable file logging.
    """
    global LOG_FILE
    LOG_FILE = file_path
    if file_path:
        log_info(f"Log file set to {file_path}", "TorchDevice.set_log_file")
    else:
        log_info("File logging disabled", "TorchDevice.set_log_file")

def get_verbosity() -> int:
    """
    Get the current log verbosity level.
    
    Returns:
        int: The current verbosity level.
    """
    return LOG_VERBOSITY
```

### 2. Update __init__.py to expose logging functions

```python
from .logging import (
    log_info, log_warning, log_error,
    set_verbosity, set_log_file, get_verbosity,
    LOG_VERBOSITY
)
```

### 3. Update References in Other Modules

All other modules that use logging functions will need to be updated to import from the new logging module:

```python
from .logging import log_info, log_warning, log_error, LOG_VERBOSITY
```

## Benefits
- Centralized logging functionality
- Easier to maintain and extend
- Better organization of code
- Improved type hints and documentation
- Additional utility functions for setting verbosity and log file
