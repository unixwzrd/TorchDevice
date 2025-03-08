# Logger for TorchDevice operations
import inspect
import logging
import os
import sys
from typing import Dict, Set


__all__ = ['log_message']


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


# Track which messages have been logged to avoid duplicates
_logged_messages: Set[str] = set()
# Maximum size for the logged messages set to prevent memory issues
_MAX_LOGGED_MESSAGES = 1000
# Track entry points to avoid internal chatter
_entry_points: Dict[str, str] = {}


def get_caller_info():
    """
    Retrieve the caller's program name, module name, class name, line number, and function name.
    Examines the call stack to find the true entry point from external code into TorchDevice.
    """
    frame = inspect.currentframe()
    outer_frames = inspect.getouterframes(frame)
    program_name = os.path.basename(sys.argv[0])
    
    # Default return if we can't find a proper caller
    default_info = {
        'program_name': program_name,
        'module_name': 'UnknownModule',
        'caller_filename': '/Users/mps/projects/AI-PROJECTS/TorchDevice/TorchDevice.py',  # Use full path
        'class_name': 'N/A',
        'caller_lineno': 0,
        'caller_func_name': 'internal',
    }
    
    # We need at least one valid frame
    if len(outer_frames) <= 1:
        return default_info
    
    # Define patterns for internal frames to skip
    internal_patterns = [
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
    
    # Identify TorchDevice frames
    torchdevice_frames = []
    for i, frame_info in enumerate(outer_frames):
        if 'TorchDevice.py' in frame_info.filename:
            torchdevice_frames.append(i)
    
    # If no TorchDevice frames found, use the immediate caller
    if not torchdevice_frames:
        frame_info = outer_frames[1] if len(outer_frames) > 1 else outer_frames[0]
        module = inspect.getmodule(frame_info.frame)
        module_name = module.__name__ if module else 'UnknownModule'
        return {
            'program_name': program_name,
            'module_name': module_name,
            'caller_filename': frame_info.filename,  # Use full path
            'class_name': 'N/A',
            'caller_lineno': frame_info.lineno,
            'caller_func_name': frame_info.function
        }
    
    # Find the highest (last) TorchDevice frame in the stack
    last_torchdevice_idx = max(torchdevice_frames) if torchdevice_frames else -1
    
    # If the last TorchDevice frame is the last frame in the stack, use default
    if last_torchdevice_idx >= len(outer_frames) - 1:
        return default_info
    
    # Start looking for user code after the last TorchDevice frame
    # If we couldn't find a TorchDevice frame, start from the second frame (skip this function)
    start_idx = last_torchdevice_idx + 1 if last_torchdevice_idx >= 0 else 1
    
    # Collect all frames after the last TorchDevice frame
    user_frames = []
    for i in range(start_idx, len(outer_frames)):
        frame_info = outer_frames[i]
        filename = frame_info.filename
        function = frame_info.function
        
        # Skip internal frames and framework infrastructure
        if any(pattern in filename for pattern in internal_patterns):
            continue
            
        # Skip unittest frames that aren't test methods
        if 'unittest' in filename and not function.startswith('test_'):
            continue
            
        # Found a user frame - collect it
        user_frames.append(i)
    
    # If we found any user frames, use the first one (closest to TorchDevice)
    if user_frames:
        frame_idx = user_frames[0]
        frame_info = outer_frames[frame_idx]
        
        # Extract information from the frame
        caller_filename = frame_info.filename  # Use full path
        lineno = frame_info.lineno
        func_name = frame_info.function
        
        # Get module name
        try:
            module = inspect.getmodule(frame_info.frame)
            module_name = module.__name__ if module else 'UnknownModule'
        except Exception:
            module_name = 'UnknownModule'
        
        # Get class name if available
        cls_name = 'N/A'
        try:
            if 'self' in frame_info.frame.f_locals:
                cls_name = frame_info.frame.f_locals['self'].__class__.__name__
            elif 'cls' in frame_info.frame.f_locals:
                cls_name = frame_info.frame.f_locals['cls'].__name__
        except Exception:
            pass
        
        return {
            'program_name': program_name,
            'module_name': module_name,
            'caller_filename': caller_filename,  # Use full path
            'class_name': cls_name,
            'caller_lineno': lineno,
            'caller_func_name': func_name
        }
    
    # If we didn't find a suitable frame, return the default
    return default_info


def log_message(message, torch_function=None):
    """Log a GPU redirection message with the given torch function."""
    # Get caller information
    info = get_caller_info()
    
    # Skip internal TorchDevice calls
    if info['caller_filename'] == 'TorchDevice.py' and info['caller_func_name'] == 'internal':
        return
    
    # Skip module-level initialization messages
    if info['caller_func_name'] == '<module>' and '__init__.py' in info['caller_filename']:
        if 'detected as default device' in message or 'Default device set to:' in message:
            return
    
    # Skip internal torch.device creation calls
    if torch_function == 'torch.device' and 'Creating torch.device' in message:
        # Only log device creation if it's explicitly requested by user code
        # Skip internal device creation for setup and initialization
        if 'setUp' in info['caller_func_name'] or 'init' in info['caller_func_name'].lower() or info['caller_func_name'] == '<module>':
            return
    
    # Skip internal initialization messages
    # Skip device detection messages in setup/init
    if 'detected as default device' in message and ('setUp' in info['caller_func_name'] or 'init' in info['caller_func_name'].lower()):
        return
    # Skip device setting messages in setup/init
    if 'Default device set to:' in message and ('setUp' in info['caller_func_name'] or 'init' in info['caller_func_name'].lower()):
        return
    # Skip tensor creation messages for internal operations
    if 'Creating tensor' in message and ('setUp' in info['caller_func_name'] or 'init' in info['caller_func_name'].lower()):
        return
    # Skip redirection messages for internal operations
    if 'Redirecting' in message and ('setUp' in info['caller_func_name'] or 'init' in info['caller_func_name'].lower()):
        return
    
    # Add torch_function to the info
    info['torch_function'] = torch_function if torch_function else 'unknown'
    
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
    is_unittest = 'unittest' in info['caller_filename'] or 'case.py' in info['caller_filename']
    is_internal_call = (
        'TorchDevice.py' in info['caller_filename'] or 
        'torch/cuda' in info['caller_filename'] or 
        'torch/backends' in info['caller_filename'] or
        'torch/_tensor.py' in info['caller_filename'] or
        'torch/nn/modules/module.py' in info['caller_filename']
    )
    
    # Skip common internal operations that generate a lot of noise
    # For redirection messages, only log the first occurrence from each location
    if 'Redirecting' in message and not is_first_call:
        return
        
    # Skip dunder method logs
    if torch_function and ('__' in torch_function or 'StreamContext' in torch_function):
        return
        
    # Skip stream operation logs
    if torch_function and 'stream' in torch_function.lower():
        return
        
    # Skip event operation logs
    if torch_function and 'event' in torch_function.lower():
        return
    
    # For unittest frames, be more selective
    if is_unittest and not info['caller_func_name'].startswith('test_'):
        # Skip unittest internals
        return
    elif is_internal_call:
        # Skip internal calls
        return
    else:
        # For user code and test methods, log if:
        # 1. It's the first time we've seen this function call from this location
        # 2. It's a new message we haven't seen before
        should_log = is_first_call or is_new_message
    
    if should_log:
        # Remember we've seen this message
        # Limit the size of the logged messages set to prevent memory issues
        if len(_logged_messages) >= _MAX_LOGGED_MESSAGES:
            # Clear the set when we reach the limit
            _logged_messages.clear()
        _logged_messages.add(message_key)
        
        # Format the message based on its content
        if message and not message.startswith('using:') and any(x in message for x in ['(', '=', ',']):
            message = f"using: {message}"
        
        # Log the message
        logger.info(message, extra=info)


# Alias for backward compatibility
log = log_message 