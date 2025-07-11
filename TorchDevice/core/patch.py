"""
TorchDevice Core Patch Module
--------------------------
Core patching functionality and orchestration.
"""

import functools
import importlib
import threading
from typing import Callable, TypeVar, Any
import torch

from .logger import log_info
from . import device  # Import device directly
from . import tensors as core_tensors
from . import modules as core_modules
from .device import DeviceManager
from .config import is_bypass_active, bypass_argument_processing, ARGUMENT_PROCESSING_EXCLUSIONS

# Thread-local guard to prevent re-entry into the tensor creation wrapper
_in_tensor_creation_wrapper = threading.local()

# Define tensor_creation_wrapper before importing ops modules to avoid circular imports
T = TypeVar('T')


def tensor_creation_wrapper(func: Callable[..., T]) -> Callable[..., T]:
    """Wrapper for tensor creation functions to enforce default device redirection and CPU override.
    Always calls torch_device_replacement for the device argument, so the toggle state is always respected.
    The device object returned by DeviceManager.torch_device_replacement is used directly.
    """
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        # If bypass is active, call the original function without redirection.
        if is_bypass_active():
            return func(*args, **kwargs)

        # Initialize thread-local if not present
        if not hasattr(_in_tensor_creation_wrapper, 'value'):
            _in_tensor_creation_wrapper.value = False

        # If we are already in a tensor creation wrapper, pass through to avoid nested logging.
        if _in_tensor_creation_wrapper.value:
            return func(*args, **kwargs)

        _in_tensor_creation_wrapper.value = True
        try:
            # Get the original device argument (if any) for logging and processing
            device_arg_original = kwargs.get('device', None)

            final_device_obj: torch.device
            if device_arg_original is None:
                # No device specified by user, get the current effective device from DeviceManager
                final_device_obj = DeviceManager.torch_device_replacement()
            else:
                # User specified a device, let DeviceManager process it (handles 'cpu:-1', redirection, etc.)
                final_device_obj = DeviceManager.torch_device_replacement(device_arg_original)

            # Check if we have a FakeDevice and handle it specially
            if hasattr(final_device_obj, '_real_device'):
                # This is a FakeDevice - create tensor on real device but wrap it
                real_device = final_device_obj._real_device
                kwargs['device'] = real_device
                result = func(*args, **kwargs)
                
                # Now we need to patch the result tensor to show the fake device type
                # This is tricky because we can't modify tensor.device directly
                # Instead, we'll create a wrapper that intercepts device property access
                class FakeDeviceTensor:
                    def __init__(self, real_tensor, fake_device):
                        self._real_tensor = real_tensor
                        self._fake_device = fake_device
                    
                    def __getattr__(self, name):
                        if name == 'device':
                            return self._fake_device
                        return getattr(self._real_tensor, name)
                    
                    def __setattr__(self, name, value):
                        if name in ['_real_tensor', '_fake_device']:
                            super().__setattr__(name, value)
                        else:
                            setattr(self._real_tensor, name, value)
                    
                    def __getitem__(self, key):
                        return self._real_tensor[key]
                    
                    def __setitem__(self, key, value):
                        self._real_tensor[key] = value
                    
                    def __str__(self):
                        return str(self._real_tensor)
                    
                    def __repr__(self):
                        return repr(self._real_tensor)
                
                return FakeDeviceTensor(result, final_device_obj)
            else:
                # Normal device object, proceed as usual
                kwargs['device'] = final_device_obj
                return func(*args, **kwargs)
        finally:
            _in_tensor_creation_wrapper.value = False
    return wrapped_func

# Import operation modules after defining the wrapper
from ..ops import (
    memory,
    nn,
    random,
    streams,
    events,
    autograd
)

# Import utility modules
from ..utils import (
    compile,
    device_utils,
    error_handling,
    type_utils
)

log_info("Initializing TorchDevice core patch module")

# Track patch status
_core_patched = False
_ops_patched = False
_utils_patched = False

# Type variable for preserving function types
T = TypeVar('T', bound=Callable[..., Any])

_original_torch_creation_functions: dict[str, Callable] = {}

# List of torch tensor creation functions to be wrapped.
# These are top-level functions in the torch module that typically accept a 'device' kwarg.
_TENSOR_CREATION_FUNCTIONS_TO_WRAP = [
    'tensor', 'as_tensor',
    'scalar_tensor',
    'ones', 'zeros', 'empty', 'full', 'eye',
    'ones_like', 'zeros_like', 'empty_like', 'full_like',
    'arange', 'range', 'linspace', 'logspace', # range is torch.range, not python range
    'rand', 'randn', 'randint',
    'empty_strided',
    # Complex tensors
    'complex', 'polar',
    # Sparse tensors (ensure wrapper handles their specific args if different)
    # 'sparse_coo_tensor', 'sparse_csr_tensor', 'sparse_csc_tensor',
    # 'sparse_bsr_tensor', 'sparse_bsc_tensor', 'sparse_compressed_tensor',
    # '_sparse_coo_tensor_unsafe', '_sparse_csr_tensor_unsafe',
]


def _apply_core_patches() -> None:
    """Apply core functionality patches."""
    global _core_patched
    if _core_patched:
        log_info("Core patches already applied")
        return

    log_info("Applying core patches...")

    # 1. Apply device patches (torch.device, torch.load)
    log_info("  Applying device.apply_patches()...")
    device.apply_patches()

    # 2. Apply tensor method patches (Tensor.to, Tensor.cuda, etc.)
    log_info("  Applying core_tensors.apply_patches()...")
    core_tensors.apply_patches()

    # 3. Apply module method patches (Module.to, Module.cuda, etc.)
    log_info("  Applying core_modules.apply_patches()...")
    core_modules.apply_patches()

    # 4. Apply tensor creation function wrappers (torch.tensor, torch.ones, etc.)
    log_info("  Applying tensor creation function wrappers...")
    for func_name in _TENSOR_CREATION_FUNCTIONS_TO_WRAP:
        if hasattr(torch, func_name):
            original_func = getattr(torch, func_name)
            if callable(original_func):
                if func_name not in _original_torch_creation_functions:
                    _original_torch_creation_functions[func_name] = original_func

                wrapped_func = tensor_creation_wrapper(original_func)
                setattr(torch, func_name, wrapped_func)
                # log_info("Patched torch.%s", func_name) # Can be verbose, enable if needed
            else:
                log_info("Skipping torch.%s as it's not callable.", func_name)
        else:
            log_info("Skipping torch.%s as it does not exist.", func_name)
    log_info("  Tensor creation function wrappers applied.")

    _core_patched = True
    log_info("Core patches application complete.")


def _apply_ops_patches() -> None:
    """Apply operation-specific patches by calling the central apply_patches of the ops package."""
    global _ops_patched
    if _ops_patched:
        log_info("Operation patches already applied")
        return

    # Import ops package here to ensure it's fully initialized before calling its apply_patches
    from .. import ops
    log_info("Applying all operation package patches via ops.apply_patches()")
    ops.apply_patches() # Call the apply_patches from TorchDevice/ops/__init__.py
    _ops_patched = True
    log_info("All operation package patches applied")


# A patch status for the bypass patches
_bypass_patched = False
_module_fallback_patched = False


def _apply_module_fallback_patches() -> None:
    """Apply module-level fallback patches for MPS compatibility issues."""
    global _module_fallback_patched
    if _module_fallback_patched:
        return

    log_info("Applying module-level fallback patches...")
    
    # Import torch.nn here to avoid circular imports
    import torch.nn as nn
    
    # Wrap LSTM forward method to handle MPS compatibility issues
    original_lstm_forward = nn.LSTM.forward
    
    @functools.wraps(original_lstm_forward)
    def lstm_forward_with_fallback(self, input, hx=None):
        try:
            return original_lstm_forward(self, input, hx)
        except RuntimeError as e:
            if ("Placeholder storage has not been allocated" in str(e) or 
                "expected device" in str(e) or 
                "indices should be either on cpu" in str(e)):
                log_info("[Module Fallback] LSTM MPS compatibility error: %s. Moving entire module to CPU.", str(e))
                
                # Store original device - handle both tensor and PackedSequence inputs
                if hasattr(input, 'device'):
                    original_device = input.device
                elif hasattr(input, 'data'):
                    original_device = input.data.device
                else:
                    # Fallback to MPS if we can't determine device
                    original_device = torch.device('mps')
                
                # Move input and hidden state to CPU
                cpu_input = input
                if hasattr(input, 'to'):
                    cpu_input = input.to('cpu')
                elif hasattr(input, 'data'):
                    # Handle PackedSequence
                    from torch.nn.utils.rnn import PackedSequence
                    cpu_input = PackedSequence(
                        input.data.to('cpu'),
                        input.batch_sizes,
                        input.sorted_indices,
                        input.unsorted_indices
                    )
                
                cpu_hx = None
                if hx is not None:
                    if isinstance(hx, tuple):
                        cpu_hx = tuple(h.to('cpu') for h in hx)
                    else:
                        cpu_hx = hx.to('cpu')
                
                # Move the entire LSTM module to CPU temporarily
                original_module_device = next(self.parameters()).device
                self.to('cpu')
                
                try:
                    # Run on CPU
                    result = original_lstm_forward(self, cpu_input, cpu_hx)
                except RuntimeError as e2:
                    # If we still get device errors even on CPU, there might be internal device issues
                    log_info("[Module Fallback] Secondary LSTM error on CPU: %s", str(e2))
                    if "Placeholder storage" in str(e2) or "indices should be either on cpu" in str(e2):
                        # Try a more aggressive approach - move everything to CPU including internal state
                        log_info("[Module Fallback] Attempting aggressive CPU fallback")
                        # Force all parameters to CPU
                        for param in self.parameters():
                            param.data = param.data.to('cpu')
                        # Force all buffers to CPU
                        for buffer in self.buffers():
                            buffer.data = buffer.data.to('cpu')
                        # Force the module to CPU and clear any cached state
                        self.to('cpu')
                        # Clear any internal state that might be cached
                        if hasattr(self, '_flat_weights'):
                            for weight in self._flat_weights:
                                if hasattr(weight, 'data'):
                                    weight.data = weight.data.to('cpu')
                        # Try again
                        result = original_lstm_forward(self, cpu_input, cpu_hx)
                    else:
                        raise
                    
                    # Move result back to original device
                    if isinstance(result, tuple):
                        output, hidden = result
                        if hasattr(output, 'to'):
                            output = output.to(original_device)
                        elif hasattr(output, 'data'):
                            # Handle PackedSequence output
                            output = PackedSequence(
                                output.data.to(original_device),
                                output.batch_sizes,
                                output.sorted_indices,
                                output.unsorted_indices
                            )
                        if isinstance(hidden, tuple):
                            hidden = tuple(h.to(original_device) for h in hidden)
                        else:
                            hidden = hidden.to(original_device)
                        return output, hidden
                    else:
                        if hasattr(result, 'to'):
                            return result.to(original_device)
                        else:
                            return result
                finally:
                    # Move module back to its original device
                    self.to(original_module_device)
            else:
                raise
    
    # Apply the patch
    nn.LSTM.forward = lstm_forward_with_fallback
    
    _module_fallback_patched = True
    log_info("Module-level fallback patches applied.")


def _apply_bypass_patches() -> None:
    """Applies bypass wrappers to functions that need to avoid device redirection."""
    global _bypass_patched
    if _bypass_patched:
        return

    log_info("Applying argument processing bypass patches...")
    for func_path in ARGUMENT_PROCESSING_EXCLUSIONS:
        try:
            module_path, func_name = func_path.rsplit('.', 1)
            # Dynamically import the module
            module = importlib.import_module(module_path)
            original_func = getattr(module, func_name)

            # Create a closure to correctly capture the original function in the loop
            def make_wrapper(func_to_wrap):
                @functools.wraps(func_to_wrap)
                def wrapper(*args, **kwargs):
                    log_info("Bypass wrapper called for %s", func_name)
                    from torch.nn.utils.rnn import PackedSequence
                    # When this function is called, bypass all argument processing.
                    with bypass_argument_processing():
                        input_device = None
                        try:
                            # For pad_packed_sequence, always move everything to CPU to avoid internal device mismatches
                            if func_name == 'pad_packed_sequence':
                                log_info("[Fallback] pad_packed_sequence detected - moving all tensors to CPU")
                                new_args = []
                                for arg in args:
                                    if isinstance(arg, torch.Tensor):
                                        new_args.append(arg.to('cpu'))
                                    else:
                                        new_args.append(arg)
                                input_device = args[0].device if args and hasattr(args[0], 'device') else None
                            else:
                                # Try on original device: handle different RNN functions
                                new_args = []
                                for i, arg in enumerate(args):
                                    if isinstance(arg, torch.Tensor):
                                        if i == 0:
                                            input_device = arg.device
                                            log_info("[Fallback] Input tensor device: %s", input_device)
                                            new_args.append(arg)
                                        elif i == 1 and func_name == 'pack_padded_sequence':
                                            # pack_padded_sequence requires lengths on CPU
                                            log_info("[Fallback] Moving lengths argument to CPU for pack_padded_sequence")
                                            new_args.append(arg.to('cpu'))
                                        else:
                                            new_args.append(arg)
                                    else:
                                        new_args.append(arg)
                            
                            result = func_to_wrap(*new_args, **kwargs)
                            # Handle return type for PackedSequence
                            if hasattr(result, 'data') and result.data.device != input_device:
                                log_info("[Fallback] Moving PackedSequence data from %s to %s", result.data.device, input_device)
                                result = PackedSequence(
                                    result.data.to(input_device),
                                    result.batch_sizes,
                                    result.sorted_indices,
                                    result.unsorted_indices
                                )
                            return result
                        except RuntimeError as e:
                            if ("expected device" in str(e) or "but got" in str(e) or "indices should be either on cpu" in str(e) or "Placeholder storage has not been allocated" in str(e)):
                                log_info("[Fallback] Device error detected: %s. Retrying on CPU.", str(e))
                                # Move all tensors to CPU for fallback
                                new_args = []
                                for i, arg in enumerate(args):
                                    if isinstance(arg, torch.Tensor):
                                        new_args.append(arg.to('cpu'))
                                    else:
                                        new_args.append(arg)
                                result = func_to_wrap(*new_args, **kwargs)
                                # Handle return type for PackedSequence
                                if hasattr(result, 'data') and input_device is not None and input_device.type != 'cpu':
                                    log_info("[Fallback] Moving PackedSequence data from CPU to %s after CPU fallback", input_device)
                                    result = PackedSequence(
                                        result.data.to(input_device),
                                        result.batch_sizes,
                                        result.sorted_indices,
                                        result.unsorted_indices
                                    )
                                return result
                            else:
                                log_info("[Fallback] Non-device error: %s", str(e))
                                raise
                return wrapper

            wrapped_func = make_wrapper(original_func)
            setattr(module, func_name, wrapped_func)
            log_info(f"  Applied bypass patch to {func_path}")
        except (ImportError, AttributeError, ValueError) as e:
            log_info(f"  Could not apply bypass patch to {func_path}: {e}")

    _bypass_patched = True
    log_info("Argument processing bypass patches applied.")


def _apply_utils_patches() -> None:
    """Apply utility patches."""
    global _utils_patched
    if _utils_patched:
        log_info("Utility patches already applied")
        return

    log_info("Applying utility patches")
    compile.apply_patches()
    device_utils.apply_patches()
    error_handling.apply_patches()
    type_utils.apply_patches()
    _utils_patched = True
    log_info("Utility patches applied")


def apply_patches() -> None:
    """
    Apply all TorchDevice patches in the correct order:
    1. Core patches (device, logging)
    2. Operation patches (memory, nn, random, streams, events, autograd)
    3. Utility patches (compile, device_utils, error_handling, type_utils)
    """
    log_info("Starting TorchDevice patch application")

    # 1. Apply bypass patches first to wrap functions that need special handling.
    _apply_bypass_patches()
    
    # 2. Apply module-level fallback patches for MPS compatibility
    _apply_module_fallback_patches()

    # 3. Core patches
    _apply_core_patches()

    # 4. Operation patches
    _apply_ops_patches()

    # 5. Utility patches
    _apply_utils_patches()

    log_info("TorchDevice patch application complete")



def ensure_patched() -> None:
    """Ensure that all patches are applied, but only once."""
    apply_patches()


__all__: list[str] = [
    'apply_patches',
    'ensure_patched',
    'tensor_creation_wrapper'
]

log_info("TorchDevice core patch module initialized")