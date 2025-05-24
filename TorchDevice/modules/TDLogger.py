import functools  # Added import for functools import logging
import logging
import os
import sys
import sysconfig

LIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STDLIB_DIR = os.path.abspath(sysconfig.get_paths()["stdlib"])

# Global flag to toggle stack frame dumping (set to True for testing/calibration)
# Use environment variable to toggle
DUMP_STACK_FRAMES = os.environ.get("DUMP_STACK_FRAMES", "False").lower() == "true"

# Add environment variable to control auto_log verbosity
LOG_LEVEL = os.environ.get("TORCHDEVICE_LOG_LEVEL", "INFO").upper()
LOG_LEVELS = {"CRITICAL": 50, "ERROR": 40, "WARNING": 30, "INFO": 20, "DEBUG": 10, "NOTSET": 0}
#LOG_LEVEL = "WARNING"

# Number of stack frames to display in debug mode.
STACK_FRAMES = 30

# You can calibrate your stack offset here once.
DEFAULT_STACK_OFFSET = 3  # adjust as needed

# Define functions to skip from logging at module level
_INTERNAL_LOG_SKIP = {
    # Core initialization and setup functions
    "apply_patches", "initialize_torchdevice", "apply_basic_patches",
    
    # Device detection and management
    "get_default_device", "redirect_device_type", "_redirect_device_type",
    
    # Tensor operations and wrappers
    "tensor_creation_wrapper", "_get_mps_event_class",
    
    # Module level functions
    "<module>", "__init__", "__main__", "__enter__", "__exit__", "__del__",
    
    # Test related functions
    "_callTestMethod", "_callSetUp", "_callTearDown",
    
    # Internal utility functions
    "wrapper", "_get_device_type", "_get_device_index"
}

def auto_log():
    """
    Decorator that logs function calls with detailed caller information.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = None
            # Only log if log level is INFO or lower
            if LOG_LEVELS.get(LOG_LEVEL, 20) <= 20 and func.__name__ not in _INTERNAL_LOG_SKIP:
                log_message(f"Called {func.__name__}", "calling the entry now")
                result = func(*args, **kwargs)
                # log_message(f"{func.__name__} returned {result}", func.__name__)
                log_message(f"{func.__name__} returned", func.__name__)
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

# Create logger and add a filter to add missing extra fields.
_logger = logging.getLogger("TorchDevice")
_handler = logging.StreamHandler(sys.stderr)
_formatter = logging.Formatter(
    'GPU REDIRECT - [%(program_name)s] "%(caller_func_name)s" in File: %(caller_filename)s:%(caller_lineno)d - '
    'Called: %(torch_function)s %(message)s'
)
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
_logger.propagate = False

# Create a separate logger for info messages
_info_logger = logging.getLogger("TorchDevice.info")
_info_handler = logging.StreamHandler(sys.stderr)
_info_formatter = logging.Formatter('INFO: [%(program_name)s] - %(message)s')
_info_handler.setFormatter(_info_formatter)
_info_logger.addHandler(_info_handler)
_info_logger.setLevel(logging.INFO)
_info_logger.propagate = False

class DefaultExtraFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'program_name'):
            record.program_name = "unknown"
        if not hasattr(record, 'caller_func_name'):
            record.caller_func_name = "unknown"
        if not hasattr(record, 'caller_filename'):
            record.caller_filename = "unknown"
        if not hasattr(record, 'caller_lineno'):
            record.caller_lineno = 0
        if not hasattr(record, 'torch_function'):
            record.torch_function = "unknown"
        return True

_logger.addFilter(DefaultExtraFilter())

def log_message(message: str, torch_function: str = "unknown", stacklevel: int = DEFAULT_STACK_OFFSET) -> None:
    """
    Log a message with detailed caller information.
    This is used primarily for GPU redirection logging.
    """
    try:
        frame = sys._getframe(stacklevel)
        caller_func_name = frame.f_code.co_name
        # Check if we need to adjust stacklevel for test methods
        if caller_func_name in ["_callTestMethod", "_callSetUp"]:
            stacklevel -= 1
            frame = sys._getframe(stacklevel)
            caller_func_name = frame.f_code.co_name
        if caller_func_name in ["wrapper"]:
            stacklevel += 1
            frame = sys._getframe(stacklevel)
            caller_func_name = frame.f_code.co_name
        if caller_func_name in ["<lambda>"]:
            stacklevel += 1
            frame = sys._getframe(stacklevel)
            caller_func_name = frame.f_code.co_name
        
        caller_filename = frame.f_code.co_filename
        caller_lineno = frame.f_lineno
    except Exception:
        caller_func_name = "unknown"
        caller_filename = "unknown"
        caller_lineno = 0

    extra = {
        "program_name": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "unknown",
        "torch_function": torch_function,
        "caller_func_name": caller_func_name,
        "caller_filename": caller_filename,
        "caller_lineno": caller_lineno,
    }
    _logger.info(message, extra=extra)

    if DUMP_STACK_FRAMES:
        dump_lines = []
        for i in range(STACK_FRAMES):
            try:
                frame = sys._getframe(i)
                formatted = f'{frame.f_code.co_name} in {os.path.abspath(frame.f_code.co_filename)}:{frame.f_lineno}'
                dump_lines.append(f'FRAME {i}: "{formatted}"')
            except ValueError:
                break
        dump = "\n".join(dump_lines)
        log_info(f"Stack frame dump:\n{dump}")
        log_info("\n**** END OF STACKFRAME DUMP ****\n\n")


def log_info(message: str) -> None:
    """
    Simple logging function that only includes the program name and message.
    This is the preferred way to log general information messages.
    
    Args:
        message: The message to log
    """
    extra = {
        "program_name": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "unknown",
    }
    _info_logger.info(message, extra=extra)

def log_warning(message: str) -> None:
    """
    Simple logging function for warnings.
    This is the preferred way to log warning messages.
    Args:
        message: The message to log
    """
    extra = {
        "program_name": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "unknown",
    }
    _info_logger.warning(message, extra=extra)
