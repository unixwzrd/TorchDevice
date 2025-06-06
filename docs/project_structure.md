# TorchDevice Project Structure

**Use the files in @TorchDevice.original as reference code as it has been tested and is working!**

## Overview
TorchDevice is organized into three main components:
1. Core functionality (device handling, patching, logging)
2. Operation patches (nn, memory, random, streams)
3. Utilities (compilation, profiling, type handling)

## Directory Structure
```
TorchDevice/
├── core/                   # Core functionality
│   ├── __init__.py
│   ├── device.py          # Base device handling
│   ├── patch.py           # Central patching orchestration & unified wrappers
│   ├── logger.py          # Logging system
│   └── tensors.py         # Tensor handling
│
├── ops/                    # Operation patches
│   ├── __init__.py
│   ├── memory/            # Memory management
│   │   ├── __init__.py
│   │   ├── management.py  # Memory allocation/deallocation
│   │   └── stats.py       # Memory usage tracking
│   │
│   ├── nn/                # Neural network operations
│   │   ├── __init__.py
│   │   ├── containers.py  # Module, Sequential, etc.
│   │   ├── layers.py      # Linear, Conv, etc.
│   │   ├── normalization.py # LayerNorm, BatchNorm
│   │   ├── activation.py  # ReLU, Sigmoid, etc.
│   │   ├── attention.py   # Attention mechanisms
│   │   └── init.py        # Weight initialization
│   │
│   ├── random/            # Random number generation
│   │   ├── __init__.py
│   │   ├── generators.py  # RNG implementations
│   │   └── distributions.py # Probability distributions
│   │
│   └── streams/           # Stream handling
│       ├── __init__.py
│       ├── cuda.py        # CUDA stream operations
│       └── mps.py         # MPS stream operations
│
├── utils/                  # Utilities and tools
│   ├── __init__.py
│   ├── compile.py         # Compilation utilities
│   ├── profiling.py       # Performance profiling
│   └── type_utils.py      # Type conversion/checking
│
├── __init__.py            # Package entry point
└── VERSION                # Version information

```

## Component Responsibilities

### Core (core/)
- **device.py**: Core device detection, selection, and management
- **patch.py**: Orchestrates all monkey-patching operations and contains unified wrapper functions
  - Includes the unified tensor_creation_wrapper for consistent device redirection
- **logger.py**: Logging system for debugging and diagnostics

### Operations (ops/)
- **memory/**: Memory management and tracking
  - Allocation/deallocation
  - Usage statistics
  - Device-specific memory operations

- **nn/**: Neural network operations
  - Layer implementations
  - Activation functions
  - Normalization
  - Attention mechanisms
  - Weight initialization

- **random/**: Random number generation
  - Comprehensive RNG implementations and patching for `torch`, `torch.cuda`, and `torch.mps`. Manages all aspects of seed setting, and RNG state get/set operations, ensuring consistent behavior across different hardware and PyTorch build configurations.
  - Probability distributions
  - Utilities for device-specific random operations, leveraging the centralized patching mechanism.

- **streams/**: Stream handling
  - CUDA stream operations
  - MPS stream operations
  - Stream synchronization

### Utilities (utils/)
- **compile.py**: Compilation and optimization
- **profiling.py**: Performance monitoring
- **type_utils.py**: Type handling and conversion

## Import Structure
```python
# Main package imports
from TorchDevice import TorchDevice, auto_log

# Core functionality
from TorchDevice.core.device import get_default_device
from TorchDevice.core.logger import log_info

# Neural network operations
from TorchDevice.ops.nn import layers, attention
from TorchDevice.ops.nn.init import xavier_uniform

# Memory operations
from TorchDevice.ops.memory import track_usage
from TorchDevice.ops.memory.stats import get_memory_stats

# Stream handling
from TorchDevice.ops.streams import get_current_stream
```

## Patching System
The patching system follows a hierarchical structure:
1. Central orchestration in `core/patch.py`
2. Category-specific patches in respective modules
3. Automatic application on package import

## Development Guidelines
1. New functionality should be placed in the appropriate category
2. Each module should have its own `apply_patches()` function
3. All patches should be registered in `core/patch.py`
4. Each module should maintain its own test suite
5. Documentation should be kept up-to-date with structure changes

## Testing Organization
Tests mirror the package structure:
```
tests/
├── core/
├── ops/
│   ├── memory/
│   ├── nn/
│   ├── random/
│   └── streams/
└── utils/
```

## Documentation
Each component maintains its own documentation:
1. Module docstrings for API documentation
2. README files for component-specific details
3. Examples in docstrings for usage patterns
4. Implementation notes in code comments 

## Migration Plan

### Phase 1: Core Infrastructure (Highest Priority)
1. **Device Management** (`core/device.py`)
   - Move device detection logic from TorchDevice.py
   - Implement device type handling and validation
   - Add device capability checks
   - Create device property wrappers

2. **Patching System** (`core/patch.py`)
   - Move patch orchestration logic
   - Implement patch registration system
   - Create patch dependency resolution
   - Add patch verification system

3. **Logging System** (`core/logger.py`)
   - Enhance logging functionality
   - Add context tracking
   - Implement log filtering
   - Add performance logging hooks

### Phase 2: Critical Operations
1. **Memory Management** (`ops/memory/`)
   - Move memory allocation/deallocation logic
   - Implement memory tracking
   - Add memory statistics
   - Create memory optimization utilities
  
2. **Random Number Generation** (`ops/random/`)
   - Move RNG implementations
   - Add distribution functions
   - Implement seed management
   - Create random utilities

3. **Stream Operations** (`ops/streams/`)
   - Complete CUDA stream implementations
   - Implement MPS stream handling
   - Add stream synchronization
   - Create stream event system

4. **Tensor Operations** (`ops/tensor/`)
   - Move tensor creation functions
   - Implement tensor conversion logic
   - Add tensor operation patches
   - Create tensor utility functions

### Phase 3: Neural Network Components
1. **Basic Layers** (`ops/nn/layers.py`)
   - Move linear layer implementations
   - Add convolution layers
   - Implement pooling layers
   - Create dropout implementations

2. **Advanced Layers** (`ops/nn/`)
   - Move attention mechanisms
   - Implement normalization layers
   - Add activation functions
   - Create initialization utilities

3. **Optimization** (`ops/optim/`)
   - Move optimizer implementations
   - Add learning rate schedulers
   - Implement gradient clipping
   - Create optimization utilities

### Phase 4: Utilities and Tools
1. **Compilation** (`utils/compile.py`)
   - Move compilation utilities
   - Add optimization passes
   - Implement caching system
   - Create build tools

2. **Profiling** (`utils/profiling.py`)
   - Move profiling tools
   - Add performance metrics
   - Implement tracing
   - Create analysis utilities

3. **Type System** (`utils/type_utils.py`)
   - Move type checking
   - Add conversion utilities
   - Implement validation
   - Create type inference tools

### Migration Guidelines
**Follow NAming Conventions and Be Consistent**
1. Each phase should be completed before moving to the next
2. Within each phase, components can be migrated in parallel if dependencies allow
3. Each migrated component must:
   - Have complete unit tests
   - Include documentation
   - Implement proper logging
   - Register its patches correctly

### Testing Strategy
1. Create unit tests before migration
2. Verify functionality during migration
3. Add integration tests after migration
4. Perform regression testing between phases

### Documentation Requirements
1. Update module docstrings
2. Add implementation notes
3. Create usage examples
4. Maintain migration status in CHANGELOG.md 