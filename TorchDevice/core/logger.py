"""
TorchDevice Core Logger Module
---------------------------
This module provides a robust, multi-level logging system for TorchDevice.
It is designed to be quiet by default while offering granular control for
developers.

Log Levels:
- WARNING (30): Default level. Shows only warnings and errors.
- INFO (20): General, high-level information about library operations.
- NOTICE (18): Detailed logs for every GPU redirection event.
- INTERNAL (15): Internal diagnostic messages for debugging TorchDevice itself.
- DEBUG (10): The most verbose output.

The log level is controlled by the `TORCHDEVICE_LOG_LEVEL` environment variable.
"""
import functools
import inspect
import logging
import os
import sys
import linecache
import threading
from typing import Any, Callable, TypeVar, ContextManager
from contextlib import contextmanager
import torch

# --- Custom Log Levels ---
NOTICE = 18
INTERNAL = 15
logging.addLevelName(NOTICE, "NOTICE")
logging.addLevelName(INTERNAL, "INTERNAL")

# --- Constants and Configuration ---
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INTERNAL": INTERNAL,
    "NOTICE": NOTICE,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


# Functions to skip in the auto_log decorator to reduce noise.
_INTERNAL_LOG_SKIP = {
    "apply_patches", "initialize_torchdevice", "apply_basic_patches",
    "get_default_device", "redirect_device_type", "_redirect_device_type",
    "tensor_creation_wrapper", "_get_mps_event_class",
    "<module>", "__init__", "__main__", "__enter__", "__exit__", "__del__",
    "_callTestMethod", "_callSetUp", "_callTearDown",
    "wrapper", "_get_device_type", "_get_device_index"
}

# --- Thread-Local State ---
_log_lock = threading.Lock()
_in_device_op = threading.local()

def _init_thread_locals() -> None:
    """Initializes thread-local variables."""
    if not hasattr(_in_device_op, 'value'):
        _in_device_op.value = False

# --- Core Logger Implementation ---

class UnbufferedStreamHandler(logging.StreamHandler):
    """A stream handler that flushes after every emit."""
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()

class TorchDeviceSimpleFormatter(logging.Formatter):
    """A simple formatter for general log messages."""
    def format(self, record: logging.LogRecord) -> str:
        return f"TorchDevice {record.levelname} - [{os.path.basename(record.filename)}:{record.lineno}] - {record.getMessage()}"

class TorchDeviceRedirectFormatter(logging.Formatter):
    """A formatter for GPU redirect logs with detailed, accurate caller info."""
    def format(self, record: logging.LogRecord) -> str:
        # Use the accurate caller info from the 'extra' dict if available
        filename = getattr(record, 'caller_filename', os.path.basename(record.pathname))
        lineno = getattr(record, 'caller_lineno', record.lineno)
        funcName = getattr(record, 'caller_funcname', record.funcName)
        pathname = getattr(record, 'caller_pathname', record.pathname)

        # Lazy formatting for the final message
        log_message = 'GPU REDIRECT - [%s] "%s" in File: %s:%s - %s'
        args = (filename, funcName, pathname, lineno, record.getMessage())
        return log_message % args


# --- Logger Setup ---

# Get log level from environment, defaulting to NOTICE
default_log_level = "NOTICE"
log_level_str = os.environ.get("TORCHDEVICE_LOG_LEVEL", default_log_level).upper()
log_level = LOG_LEVEL_MAP.get(log_level_str, NOTICE)

# 1. General-purpose logger
_logger = logging.getLogger("TorchDevice.general")
_logger.setLevel(log_level)
_logger.propagate = False

# 2. GPU Redirect logger
_redirect_logger = logging.getLogger("TorchDevice.redirect")
_redirect_logger.setLevel(NOTICE)
_redirect_logger.propagate = False

_general_handler = None
_redirect_handler = None
_redirect_file_handler = None

def _setup_handlers():
    """Create and configure handlers for both loggers."""
    global _general_handler, _redirect_handler

    # 1. General-purpose logger
    if not _logger.handlers:
        _general_handler = UnbufferedStreamHandler(sys.stderr)
        _general_handler.setFormatter(TorchDeviceSimpleFormatter())
        _general_handler.setLevel(log_level)
        _logger.addHandler(_general_handler)

    # 2. GPU Redirect logger
    if not _redirect_logger.handlers:
        _redirect_handler = UnbufferedStreamHandler(sys.stderr)
        _redirect_handler.setFormatter(TorchDeviceRedirectFormatter())
        _redirect_logger.addHandler(_redirect_handler)


# --- Test Suite Interface ---
def set_redirect_log_stream(stream=None):
    """Adds or removes a file handler for the redirect logger. For testing only."""
    global _redirect_file_handler
    # Remove existing file handler if it exists
    if _redirect_file_handler:
        _redirect_logger.removeHandler(_redirect_file_handler)
        _redirect_file_handler = None

    # Add a new handler if a stream is provided
    if stream:
        _redirect_file_handler = UnbufferedStreamHandler(stream)
        _redirect_file_handler.setFormatter(TorchDeviceRedirectFormatter())
        _redirect_logger.addHandler(_redirect_file_handler)


# --- Public Logging Functions ---
def log_internal(message: str, *args: Any) -> None:
    """Logs a message at the INTERNAL level."""
    _logger.log(INTERNAL, message, *args, stacklevel=2)

def log_info(message: str, *args: Any) -> None:
    """Logs a message at the INTERNAL level."""
    _logger.log(INTERNAL, message, *args, stacklevel=2)

def log_warning(message: str, *args: Any) -> None:
    """Logs a message at the WARNING level."""
    _logger.warning(message, *args, stacklevel=2)

def log_error(message: str, *args: Any) -> None:
    """Logs a message at the ERROR level."""
    _logger.error(message, *args, stacklevel=2)

# --- Helper Functions for Decorator ---
F = TypeVar('F', bound=Callable[..., Any])

def _is_tensor(obj: Any) -> bool:
    return isinstance(obj, torch.Tensor)

def _safe_repr(obj: Any) -> str:
    if _is_tensor(obj):
        return f"Tensor(shape={obj.shape}, device='{obj.device}')"
    try:
        return repr(obj)
    except Exception:
        return f"<repr-error: {type(obj).__name__}>"

def _truncate_long_string(s: str, context: str, max_lines: int = 15, head_size: int = 5, tail_size: int = 5) -> str:
    """Truncates a multi-line string if it exceeds max_lines."""
    lines = s.splitlines()
    if len(lines) > max_lines:
        head = lines[:head_size]
        tail = lines[-tail_size:]
        num_truncated = len(lines) - (head_size + tail_size)

        truncated_s = '\n'.join(head)
        truncated_s += f'\n... [ {num_truncated} lines of {context} removed from middle ] ...\n'
        truncated_s += '\n'.join(tail)
        return truncated_s
    return s

def _format_call_args(func: Callable, *args: Any, **kwargs: Any) -> str:
    """Formats the function's arguments into a string, truncating long ones."""
    try:
        # Use bind_partial to handle methods where 'self' is implicitly passed
        bound_args = inspect.signature(func).bind_partial(*args, **kwargs)
        arg_strs = []
        for name, value in bound_args.arguments.items():
            # Skip 'self' and 'cls' for cleaner logs
            if name in ('self', 'cls'):
                continue
            arg_strs.append(f"{name}={_truncate_long_string(_safe_repr(value), 'ARGUMENTS')}")
        return f"{func.__name__}({', '.join(arg_strs)})"
    except (ValueError, TypeError):
        # Fallback for complex callables where binding fails
        arg_strs = [_truncate_long_string(_safe_repr(arg), 'ARGUMENTS') for arg in args]
        kwarg_strs = [f"{k}={_truncate_long_string(_safe_repr(v), 'ARGUMENTS')}" for k, v in kwargs.items()]
        return f"{func.__name__}({', '.join(arg_strs + kwarg_strs)})"

def _format_result(result: Any) -> str:
    """Formats the function's result into a string, truncating it if it's too long."""
    return _truncate_long_string(_safe_repr(result), 'RESULT')


# --- Public Context Manager ---
@contextmanager
def device_operation_context() -> ContextManager[None]:
    _init_thread_locals()
    original_value = _in_device_op.value
    _in_device_op.value = True
    try:
        yield
    finally:
        _in_device_op.value = original_value

# --- Decorator ---
def auto_log() -> Callable[[F], F]:
    """Decorator to log function calls with their arguments and results."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _init_thread_locals()
            if _in_device_op.value or func.__name__ in _INTERNAL_LOG_SKIP:
                return func(*args, **kwargs)

            _in_device_op.value = True
            try:
                result = func(*args, **kwargs)

                if _redirect_logger.isEnabledFor(NOTICE):
                    # --- Find the true caller frame ---
                    frame = inspect.currentframe()
                    torchdevice_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    caller_frame = None
                    try:
                        while frame:
                            frame_filename = os.path.abspath(frame.f_code.co_filename)
                            func_name = frame.f_code.co_name
                            if not frame_filename.startswith(torchdevice_root_path) and func_name not in _INTERNAL_LOG_SKIP:
                                caller_frame = frame
                                break
                            frame = frame.f_back
                    finally:
                        if frame: del frame

                    # --- Format and Log ---
                    formatted_call = _format_call_args(func, args, kwargs)
                    formatted_result = _format_result(result)

                    if caller_frame:
                        # Get the source code line for more descriptive logging
                        try:
                            filename = caller_frame.f_code.co_filename
                            lineno = caller_frame.f_lineno
                            source_line = linecache.getline(filename, lineno).strip()
                            if not source_line: # Fallback for empty lines
                                source_line = caller_frame.f_code.co_name
                        except Exception:
                            source_line = caller_frame.f_code.co_name # Final fallback

                        extra_info = {
                            'caller_filename': os.path.basename(caller_frame.f_code.co_filename),
                            'caller_lineno': caller_frame.f_lineno,
                            'caller_funcname': source_line, # Use the source line instead of just the function name
                            'caller_pathname': os.path.abspath(caller_frame.f_code.co_filename)
                        }
                        _redirect_logger.log(
                            NOTICE,
                            'CALLED: %s | RETURNED: %s',
                            formatted_call,
                            formatted_result,
                            extra=extra_info
                        )
                    else:
                        # Fallback if a caller frame couldn't be found
                        _redirect_logger.log(
                            NOTICE,
                            'CALLED: %s | RETURNED: %s',
                            formatted_call,
                            formatted_result
                        )

                return result
            finally:
                _in_device_op.value = False
        return wrapper
    return decorator

# --- Module Initialization ---
_setup_handlers()  # Initial setup
_init_thread_locals()
log_info("TorchDevice logger initialized. Log Level: %s", log_level_str)

__all__ = [
    'log_internal', 'log_info', 'log_warning', 'log_error',
    'device_operation_context', 'auto_log', 'NOTICE', 'INTERNAL',
    'set_redirect_log_stream' # Exposed for test suite
]
