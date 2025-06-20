# TorchDevice API Reference

This document provides a comprehensive reference for all TorchDevice modules, functions, and their current implementation status.

## Core Module (`core/`)

The core module provides fundamental device handling and system integration.

### Device Management (`core/device.py`)
```python
def get_device(device_str: str) -> torch.device
def set_default_device(device: torch.device) -> None
def get_current_device() -> torch.device
```

### Patching System (`core/patch.py`)
```python
def patch_torch() -> None
def unpatch_torch() -> None
def is_patched() -> bool
```

### Logging System (`core/logger.py`)
```python
def log_message(msg: str, level: str = "INFO") -> None
def set_log_level(level: str) -> None
def get_log_level() -> str
```

## Operations Module (`ops/`)

### Memory Management (`ops/memory/`)
```python
# management.py
def empty_cache() -> None
def memory_stats() -> Dict[str, int]
def memory_allocated() -> int
def memory_reserved() -> int

# stats.py
def get_memory_summary() -> str
def track_memory_usage() -> ContextManager
```

### Neural Networks (`ops/nn/`)
```python
# layers.py
class Linear(nn.Module): ...
class Conv2d(nn.Module): ...

# normalization.py
class LayerNorm(nn.Module): ...
class BatchNorm2d(nn.Module): ...

# activation.py
class ReLU(nn.Module): ...
class GELU(nn.Module): ...

# attention.py
class MultiHeadAttention(nn.Module): ...
class SelfAttention(nn.Module): ...

# init.py
def xavier_uniform_(tensor: Tensor) -> Tensor
def kaiming_normal_(tensor: Tensor) -> Tensor
```

### Random Number Generation (`ops/random/`)
```python
# generators.py
def manual_seed(seed: int) -> None
def randn(*size: int, device: Optional[torch.device] = None) -> Tensor
def rand(*size: int, device: Optional[torch.device] = None) -> Tensor

# distributions.py
class Normal: ...
class Uniform: ...
```

### Stream Management (`ops/streams/`)
```python
# cuda.py
class Stream: ...
def current_stream() -> Stream
def default_stream() -> Stream

# mps.py
class MPSStream: ...
def mps_current_stream() -> MPSStream

# synchronize.py
def synchronize_stream(stream: Stream) -> None
def wait_stream(stream: Stream) -> None
```

### Event Management (`ops/events/`)
```python
# cuda_events.py
class CUDAEvent: ...
def record_event(event: CUDAEvent, stream: Optional[Stream] = None) -> None

# mps_events.py
class MPSEvent: ...
def record_mps_event(event: MPSEvent, stream: Optional[MPSStream] = None) -> None

# synchronize.py
def synchronize_event(event: Union[CUDAEvent, MPSEvent]) -> None
def wait_event(event: Union[CUDAEvent, MPSEvent]) -> None
```

### Automatic Differentiation (`ops/autograd/`)
```python
# function.py
class Function: ...
def register_backward_hook(hook: Callable) -> None

# variable.py
class Variable: ...
def detach(tensor: Tensor) -> Tensor

# grad_mode.py
def set_grad_enabled(mode: bool) -> ContextManager
def no_grad() -> ContextManager
def enable_grad() -> ContextManager
```

### Optimization (`ops/optim/`)
```python
# optimizer.py
class Optimizer: ...
class SGD(Optimizer): ...
class Adam(Optimizer): ...

# lr_scheduler.py
class LRScheduler: ...
class StepLR(LRScheduler): ...
class CosineAnnealingLR(LRScheduler): ...
```

## Utilities Module (`utils/`)

### Compilation (`utils/compile.py`)
```python
def compile_model(model: nn.Module) -> nn.Module
def is_compiled(model: nn.Module) -> bool
```

### Profiling (`utils/profiling.py`)
```python
def profile_execution(callable: Callable) -> Dict[str, float]
def memory_profiler(callable: Callable) -> Dict[str, int]
```

### Type Utilities (`utils/type_utils.py`)
```python
def check_device_type(device: torch.device) -> bool
def validate_dtype(dtype: torch.dtype) -> bool
```

### Device Utilities (`utils/device_utils.py`)
```python
def get_device_info() -> Dict[str, Any]
def check_device_compatibility() -> bool
```

### Error Handling (`utils/error_handling.py`)
```python
def handle_device_error(error: Exception) -> None
class DeviceError(Exception): ...
```

## Implementation Status

| Module | Status | Test Coverage | Notes |
|--------|--------|---------------|-------|
| core/device | ✓ Complete | ✓ Full | Core functionality |
| core/patch | ✓ Complete | ✓ Full | Patching system |
| core/logger | ✓ Complete | ✓ Full | Logging system |
| ops/memory | ✓ Complete | ✓ Full | Memory management |
| ops/nn | ✓ Complete | ✓ Full | Neural network ops |
| ops/random | ✓ Complete | ✓ Full | RNG functionality |
| ops/streams | ✓ Complete | ✓ Full | Stream management |
| ops/events | ⚠ In Progress | ⚠ Partial | Event handling |
| ops/autograd | ⚠ In Progress | ⚠ Partial | Automatic differentiation |
| ops/optim | ⚠ In Progress | ⚠ Partial | Optimization algorithms |
| utils/* | ✓ Complete | ✓ Full | Utility functions |

**Status Legend:**
- ✓ Complete: Fully implemented and tested
- ⚠ In Progress: Partially implemented or under development
- ✗ Planned: Not yet implemented

**Test Coverage:**
- ✓ Full: Comprehensive test suite
- ⚠ Partial: Some tests implemented
- ✗ None: No tests yet

## Usage Examples

See the [examples/](../examples/) directory for detailed usage examples of each module. 