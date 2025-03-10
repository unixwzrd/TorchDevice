# Logger for TorchDevice operations
import inspect
import logging
import os
import sys
import collections
from typing import Dict, Deque, Any, Optional


__all__ = ['log_message']


# Configure logger
logger = logging.getLogger('TorchDevice')
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(
    'GPU REDIRECT - [%(program_name)s] "%(caller_func_name)s" in File: %(caller_filename)s:%(caller_lineno)s - '
    'Called: %(torch_function)s %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
# Prevent propagation to avoid duplicate messages
logger.propagate = False


# Maximum size for the logged messages collection to prevent memory issues
_MAX_LOGGED_MESSAGES = 1000
# Track which messages have been logged to avoid duplicates
# Using a deque with maxlen automatically removes oldest entries when full
_logged_messages: Deque[str] = collections.deque(maxlen=_MAX_LOGGED_MESSAGES)
# Track entry points to avoid internal chatter
_entry_points: Dict[str, str] = {}

# Constants for filtering
# Patterns for internal frames to skip
INTERNAL_PATTERNS = [
    'TorchDevice.py', 
    'inspect.py', 
    'case.py', 
    'suite.py', 
    'runner.py',
    'torch/cuda',
    'torch/backends',
    'torch/_tensor.py',
    'torch/nn/modules/module.py'
]

# Messages to skip during setup/initialization
SETUP_INIT_MESSAGES = [
    'detected as default device',
    'Default device set to:',
    'Creating tensor',
    'Redirecting'
]

# Function types to skip
SKIP_FUNCTION_TYPES = [
    'stream',
    'event'
]

# Messages that are always important and should never be skipped
IMPORTANT_MESSAGE_PATTERNS = [
    'Redirecting',
    'Synchronizing'
]


def is_test_environment() -> bool:
    """
    Determine if the current environment is a test environment.
    
    Returns:
        bool: True if running in a test environment, False otherwise.
    """
    # Check if running through unittest or pytest
    if os.path.basename(sys.argv[0]) in ['unittest', 'pytest']:
        return True
    
    # Check if the program name contains test indicators
    program_name = os.path.basename(sys.argv[0])
    if any(pattern in program_name for pattern in ['test', 'run_tests', 'unittest']):
        return True
    
    # Check if any frame in the stack is from a test file
    frame = inspect.currentframe()
    if frame:
        outer_frames = inspect.getouterframes(frame)
        for frame_info in outer_frames:
            if 'test' in frame_info.filename.lower():
                return True
    
    return False


def is_internal_frame(filename: str, function_name: str) -> bool:
    """
    Determine if a frame is an internal frame that should be skipped.
    
    Args:
        filename: The filename of the frame
        function_name: The function name of the frame
        
    Returns:
        bool: True if the frame is internal, False otherwise
    """
    # Check if the filename matches any internal pattern
    if any(pattern in filename for pattern in INTERNAL_PATTERNS):
        return True
    
    # Skip unittest frames that aren't test methods
    if 'unittest' in filename and not function_name.startswith('test_'):
        return True
    
    return False


def is_setup_or_init(func_name: str) -> bool:
    """
    Determine if a function name is related to setup or initialization.
    
    Args:
        func_name: The function name to check
        
    Returns:
        bool: True if the function is related to setup or initialization, False otherwise
    """
    return (
        'setUp' in func_name or 
        'init' in func_name.lower() or 
        func_name == '<module>'
    )


def contains_important_message(message: str) -> bool:
    """
    Determine if a message contains important information that should never be skipped.
    
    Args:
        message: The message to check
        
    Returns:
        bool: True if the message contains important information, False otherwise
    """
    return any(pattern in message for pattern in IMPORTANT_MESSAGE_PATTERNS)


def should_skip_message(info: Dict[str, Any], message: str, torch_function: Optional[str]) -> bool:
    """
    Determine if a message should be skipped based on various filtering criteria.
    
    Args:
        info: The caller information dictionary
        message: The log message
        torch_function: The torch function name
        
    Returns:
        bool: True if the message should be skipped, False otherwise
    """
    # Never skip important messages like redirections and synchronizations
    if contains_important_message(message):
        return False
    
    # Skip internal TorchDevice calls
    if info['caller_filename'] == 'TorchDevice.py' and info['caller_func_name'] == 'internal':
        return True
    
    # Skip module-level initialization messages
    if info['caller_func_name'] == '<module>' and '__init__.py' in info['caller_filename']:
        if any(msg in message for msg in SETUP_INIT_MESSAGES):
            return True
    
    # Skip internal torch.device creation calls during setup/init
    if torch_function == 'torch.device' and 'Creating torch.device' in message and is_setup_or_init(info['caller_func_name']):
        return True
    
    # Skip internal initialization messages for setup/init functions
    if is_setup_or_init(info['caller_func_name']):
        # Skip device detection and setting messages in setup/init
        if any(msg in message for msg in SETUP_INIT_MESSAGES[:2]):  # Only use the first two setup messages
            return True
        # Skip tensor creation messages for internal operations
        if 'Creating tensor' in message:
            return True
    
    # Skip dunder method logs
    if torch_function and ('__' in torch_function or 'StreamContext' in torch_function):
        return True
    
    # Skip specific function type logs
    if torch_function:
        for func_type in SKIP_FUNCTION_TYPES:
            if func_type in torch_function.lower():
                return True
    
    # For unittest frames, be more selective
    is_unittest = 'unittest' in info['caller_filename'] or 'case.py' in info['caller_filename']
    if is_unittest and not info['caller_func_name'].startswith('test_'):
        return True
    
    # Check if it's an internal call
    is_internal_call = any(pattern in info['caller_filename'] for pattern in INTERNAL_PATTERNS)
    if is_internal_call:
        return True
    
    return False


def get_outer_frames() -> list:
    """Return the outer frames once, to avoid multiple calls to inspect.getouterframes."""
    frame = inspect.currentframe()
    return inspect.getouterframes(frame)


def collect_torchdevice_indices(outer_frames: list) -> list:
    """Return a list of indices for frames whose filename contains 'TorchDevice.py'."""
    return [i for i, fi in enumerate(outer_frames) if 'TorchDevice.py' in fi.filename]


def find_user_frame(outer_frames: list, torchdevice_indices: list) -> Optional[inspect.FrameInfo]:
    """
    Return the first frame after the last TorchDevice frame that isn't internal.
    """
    start_idx = torchdevice_indices[-1] + 1 if torchdevice_indices else 1
    for fi in outer_frames[start_idx:]:
        if not is_internal_frame(fi.filename, fi.function):
            return fi  # type: ignore
    # Fallback: return the first frame not in TorchDevice
    for fi in outer_frames:
        if 'TorchDevice.py' not in fi.filename:
            return fi  # type: ignore
    return None


def format_caller_info(frame_info: inspect.FrameInfo, program_name: str) -> Dict[str, Any]:
    """Return a dictionary with caller info based on a frame record."""
    module = inspect.getmodule(frame_info.frame)
    module_name = module.__name__ if module else 'UnknownModule'
    cls_name = 'N/A'
    if 'self' in frame_info.frame.f_locals:
        cls_name = frame_info.frame.f_locals['self'].__class__.__name__
    elif 'cls' in frame_info.frame.f_locals:
        cls_name = frame_info.frame.f_locals['cls'].__name__
    return {
        'program_name': program_name,
        'module_name': module_name,
        'caller_filename': frame_info.filename,
        'class_name': cls_name,
        'caller_lineno': frame_info.lineno,
        'caller_func_name': frame_info.function,
    }


def get_caller_info() -> Dict[str, Any]:
    """Retrieve the caller information using a single pass through the stack."""
    outer_frames = get_outer_frames()
    # Determine program name based on sys.argv[0] and frame filenames.
    program_name = "TorchDevice"
    if os.path.basename(sys.argv[0]) in ['unittest', 'pytest']:
        program_name = "test"
    else:
        program_name = os.path.basename(sys.argv[0])
        if any(p in program_name for p in ['test', 'run_tests', 'unittest']):
            program_name = "test"
        for fi in outer_frames:
            if 'test' in fi.filename.lower():
                program_name = "test"
                break

    torchdevice_indices = collect_torchdevice_indices(outer_frames)
    candidate = find_user_frame(outer_frames, torchdevice_indices)
    if candidate is not None:
        return format_caller_info(candidate, program_name)
    else:
        # Fallback default info.
        return {
            'program_name': program_name,
            'module_name': 'UnknownModule',
            'caller_filename': 'unknown',
            'class_name': 'N/A',
            'caller_lineno': 0,
            'caller_func_name': 'unknown'
        }


def log_message(message, torch_function=None):
    """Log a GPU redirection message with the given torch function."""
    try:
        # Get caller information
        info = get_caller_info()
        
        # When log_message is called directly (not from TorchDevice redirections)
        # Use the provided function name instead of relying solely on stack inspection
        if torch_function and 'torch_function' not in info:
            # This means log_message was called directly with a specific function name
            # Use that function name directly in the logging
            info['torch_function'] = torch_function
        else:
            # Add torch_function to the info for normal redirections
            info['torch_function'] = torch_function if torch_function else 'unknown'
        
        # Check if we should skip this message
        if should_skip_message(info, message, torch_function):
            return
        
        # Create a unique key for this message to avoid duplicates
        caller_key = f"{info['caller_filename']}:{info['caller_func_name']}:{info['caller_lineno']}"
        function_key = torch_function if torch_function else 'unknown'
        message_key = f"{caller_key}:{function_key}:{message}"
        
        # Track the first time we see a torch function call from a specific location
        # Only track actual user code, not unittest internals
        if function_key not in _entry_points and 'unittest' not in info['caller_filename'] and 'case.py' not in info['caller_filename']:
            _entry_points[function_key] = caller_key
        
        # Determine if we should log this message
        is_first_call = _entry_points.get(function_key) == caller_key
        is_new_message = message_key not in _logged_messages
        
        # For important messages or first occurrences, log the message
        should_log = contains_important_message(message) or is_first_call or is_new_message
        
        if should_log:
            # Remember we've seen this message
            _logged_messages.append(message_key)
            
            # Log the message
            logger.info(message, extra=info)
    except Exception as e:
        # Ensure exceptions in the logger don't propagate to the main application
        print(f"Error in TDLogger: {str(e)}", file=sys.stderr)
