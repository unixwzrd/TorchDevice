"""
TorchDevice Core Logger Module
---------------------------
Core logging functionality with a focus on GPU redirection.
"""

import functools
import logging
import sys
import os
import threading
from typing import Any, Callable, TypeVar, Optional, ContextManager
from contextlib import contextmanager

# --- Constants based on TDLogger.py and user rules ---
STACK_FRAMES = 30
DEFAULT_STACK_OFFSET = 3
DUMP_STACK_FRAMES = os.environ.get("TORCHDEVICE_DUMP_STACK_FRAMES", "False").lower() == "true"
LOG_LEVEL_ENV = os.environ.get("TORCHDEVICE_LOG_LEVEL", "INFO").upper()
LOG_LEVELS = {"CRITICAL": 50, "ERROR": 40, "WARNING": 30, "INFO": 20, "DEBUG": 10, "NOTSET": 0}
CURRENT_LOG_LEVEL_INT = LOG_LEVELS.get(LOG_LEVEL_ENV, 20)

_INTERNAL_LOG_SKIP = {
    "apply_patches", "initialize_torchdevice", "apply_basic_patches",
    "get_default_device", "redirect_device_type", "_redirect_device_type",
    "tensor_creation_wrapper", "_get_mps_event_class",
    "<module>", "__init__", "__main__", "__enter__", "__exit__", "__del__",
    "_callTestMethod", "_callSetUp", "_callTearDown",
    "wrapper", "_get_device_type", "_get_device_index",
    "_safe_repr", "_is_tensor", "log_message", "log_info", "log_warning", "log_error",
    "device_operation_context"
}
# --- End Constants ---

# --- Logger Setup ---
_redirect_logger = logging.getLogger("TorchDevice.Redirect")
_redirect_handler = logging.StreamHandler(sys.stderr)
_redirect_formatter = logging.Formatter(
    'GPU REDIRECT - [%(program_name)s] "%(caller_func_name)s" in File: %(caller_filename)s:%(caller_lineno)d - '
    'Called: %(torch_function)s %(message)s'
)
_redirect_handler.setFormatter(_redirect_formatter)
_redirect_logger.addHandler(_redirect_handler)
_redirect_logger.setLevel(logging.INFO) # GPU Redirect logs are always INFO level if enabled
_redirect_logger.propagate = False

_info_logger = logging.getLogger("TorchDevice.Info")
_info_handler = logging.StreamHandler(sys.stdout) # General logs to stdout
_info_formatter = logging.Formatter('[TORCHDEVICE] [%(levelname)s] %(message)s')
_info_handler.setFormatter(_info_formatter)
_info_logger.addHandler(_info_handler)
_info_logger.setLevel(CURRENT_LOG_LEVEL_INT) # General log level controlled by env var
_info_logger.propagate = False

class DefaultExtraFilter(logging.Filter):
    def filter(self, record):
        record.program_name = getattr(record, 'program_name', os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "unknown")
        record.caller_func_name = getattr(record, 'caller_func_name', "unknown")
        record.caller_filename = getattr(record, 'caller_filename', "unknown")
        record.caller_lineno = getattr(record, 'caller_lineno', 0)
        record.torch_function = getattr(record, 'torch_function', "unknown")
        return True


_redirect_logger.addFilter(DefaultExtraFilter())
_info_logger.addFilter(DefaultExtraFilter())
# --- End Logger Setup ---

_log_lock = threading.Lock()
_logging_depth = threading.local()
_in_device_op = threading.local()

def _init_thread_locals():
    """Initializes thread-local storage if not already done."""
    if not hasattr(_logging_depth, 'value'):
        _logging_depth.value = 0
    if not hasattr(_in_device_op, 'value'):
        _in_device_op.value = False

F = TypeVar('F', bound=Callable[..., Any])

@contextmanager
def device_operation_context() -> ContextManager[None]:
    """Context manager to temporarily disable logging, e.g., during sensitive device ops."""
    _init_thread_locals()
    was_in_device_op = _in_device_op.value
    _in_device_op.value = True
    try:
        yield
    finally:
        _in_device_op.value = was_in_device_op

def _is_tensor(obj: Any) -> bool:
    """Checks if an object is a PyTorch tensor without triggering its __str__ or __repr__ if it's not needed."""
    try:
        # Avoid importing torch at module level if not already imported
        if 'torch' in sys.modules:
            return sys.modules['torch'].is_tensor(obj)
        return False # If torch not imported, can't be a torch tensor
    except Exception:
        return False

def _safe_repr(obj: Any) -> str:
    """Safely convert an object to its string representation, handling tensors specially."""
    if _is_tensor(obj):
        try:
            # Using f-string for lazy evaluation if obj.shape or obj.device access fails
            return f"<Tensor shape={list(obj.shape)} device={obj.device.type} dtype={obj.dtype}>"
        except Exception:
            return "<Tensor (error getting details)>"
    try:
        s = str(obj)
        return s if len(s) < 200 else s[:197] + "..." # Truncate long strings
    except Exception:
        return f"<{type(obj).__name__} (unprintable)>"

def log_message(message: str, torch_function: str = "unknown", stacklevel_offset: int = 0) -> None:
    """Logs a GPU REDIRECT message with detailed caller context."""
    _init_thread_locals()
    # GPU redirect messages are only logged if global log level is INFO or more verbose
    if _in_device_op.value or CURRENT_LOG_LEVEL_INT > logging.INFO:
        return

    actual_stacklevel = DEFAULT_STACK_OFFSET + stacklevel_offset
    try:
        frame = sys._getframe(actual_stacklevel)
        caller_func_name = frame.f_code.co_name
        # Traverse up the stack if current frame is an internal wrapper/decorator
        while caller_func_name in ['wrapper', '<lambda>', '_inner', 'decorator', 'auto_log_wrapper'] and actual_stacklevel < STACK_FRAMES:
             actual_stacklevel += 1
             frame = sys._getframe(actual_stacklevel)
             caller_func_name = frame.f_code.co_name
        caller_filename = frame.f_code.co_filename
        caller_lineno = frame.f_lineno
    except ValueError: # Reached top or bottom of stack
        caller_func_name = "unknown (deep stack)"
        caller_filename = "unknown"
        caller_lineno = 0
    except Exception: # Any other error getting frame info
        caller_func_name = "unknown (frame error)"
        caller_filename = "unknown"
        caller_lineno = 0

    extra = {
        "program_name": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "script",
        "torch_function": torch_function,
        "caller_func_name": caller_func_name,
        "caller_filename": caller_filename,
        "caller_lineno": caller_lineno,
    }
    with _log_lock: # Ensure thread safety for logging
        _redirect_logger.info(message, extra=extra)

    if DUMP_STACK_FRAMES:
        dump_lines = ["Stack frame dump (for log_message):"]
        # Start dump from one level above the current log_message call
        # The frame for log_message itself is actual_stacklevel - 1 (relative to its caller)
        # So, start dump from actual_stacklevel
        dump_start_frame_depth = actual_stacklevel
        for i in range(STACK_FRAMES):
            try:
                frame_dump = sys._getframe(dump_start_frame_depth + i)
                formatted = f'  FRAME {i}: {frame_dump.f_code.co_name} in {os.path.abspath(frame_dump.f_code.co_filename)}:{frame_dump.f_lineno}'
                dump_lines.append(formatted)
            except ValueError:
                break # No more frames
        _info_logger.debug("\n".join(dump_lines)) # Use _info_logger at DEBUG for stack dumps

def log_info(message: str, *args: Any) -> None:
    """Logs a general informational message using the TORCHDEVICE.Info logger."""
    _init_thread_locals()
    if _in_device_op.value or CURRENT_LOG_LEVEL_INT > logging.INFO:
        return
    with _log_lock:
        _info_logger.info(message % args if args else message)

def log_warning(message: str, *args: Any) -> None:
    """Logs a warning message using the TORCHDEVICE.Info logger."""
    _init_thread_locals()
    if _in_device_op.value or CURRENT_LOG_LEVEL_INT > logging.WARNING:
        return
    with _log_lock:
        _info_logger.warning(message % args if args else message)

def log_error(message: str, *args: Any) -> None:
    """Logs an error message using the TORCHDEVICE.Info logger."""
    # Errors are always logged unless in device_op_context
    _init_thread_locals()
    if _in_device_op.value:
        return
    with _log_lock:
        _info_logger.error(message % args if args else message)

def auto_log() -> Callable[[F], F]:
    """Decorator factory to automatically log function calls using `log_message` for GPU redirection context."""
    def decorator(func: F) -> F:
        func_qualname = getattr(func, '__qualname__', func.__name__)
        if func_qualname in _INTERNAL_LOG_SKIP or func.__name__ in _INTERNAL_LOG_SKIP:
            return func # Do not wrap functions in the skip list

        @functools.wraps(func)
        def auto_log_wrapper(*args: Any, **kwargs: Any) -> Any:
            _init_thread_locals()
            # Check if logging for this level is enabled. INFO is used by auto_log implicitly.
            # The original check was `_in_device_op.value or CURRENT_LOG_LEVEL_INT > logging.INFO`
            # which meant auto_log messages (which are INFO level) would be skipped if global level was higher than INFO.
            # And also if _in_device_op.value was true (internal op, skip detailed logging).
            # Let's refine this: auto_log messages are INFO. They should appear if global level is INFO or DEBUG.
            if _in_device_op.value or CURRENT_LOG_LEVEL_INT > logging.INFO:
                 return func(*args, **kwargs)

            _logging_depth.value += 1
            # Limit recursion depth for auto_log to prevent excessive logging
            if _logging_depth.value > 10:
                result = func(*args, **kwargs)
                _logging_depth.value -=1
                return result

            effective_func_name = func.__name__ # original_func_name_override is removed

            # Lazy formatting of args for the log message
            # The full string is only constructed if logging actually happens.
            # Using lazy % formatting as per developer rules for the final log message.
            args_repr = "()"
            if args or kwargs:
                arg_strs = [_safe_repr(arg) for arg in args]
                kwarg_strs = [f"{k}={_safe_repr(v)}" for k, v in kwargs.items()]
                args_repr = f"({', '.join(arg_strs + kwarg_strs)})"

            # Pass 1 to stacklevel_offset because log_message is called from within this wrapper
            log_message("called with args: %s" % args_repr, torch_function=effective_func_name, stacklevel_offset=1)

            try:
                result = func(*args, **kwargs)
                log_message("returned: %s" % _safe_repr(result), torch_function=effective_func_name, stacklevel_offset=1)
                return result
            except Exception as e:
                log_message("raised: %s: %s" % (type(e).__name__, _safe_repr(e)), torch_function=effective_func_name, stacklevel_offset=1)
                raise
            finally:
                _logging_depth.value -= 1
        return auto_log_wrapper
    return decorator

# Initial log message to confirm logger is active and show configured log level
log_info("TorchDevice core logger module initialized. GPU Redirect Log Level: INFO. General Log Level: %s", LOG_LEVEL_ENV)

__all__ = [
    'log_message', 'log_info', 'log_warning', 'log_error',
    'auto_log', 'device_operation_context',
    'DUMP_STACK_FRAMES', 'CURRENT_LOG_LEVEL_INT', 'LOG_LEVEL_ENV'
]
