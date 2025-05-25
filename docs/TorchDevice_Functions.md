# TorchDevice API Reference

This document tracks all functions and modules in TorchDevice, including their implementation status, location, and test coverage.

## Core Functionality

| Function Name                | Description                                | Status      | Module Location              | Test Coverage |
|:----------------------------|:------------------------------------------|:------------|:----------------------------|:-------------:|
| **Device Management**       |                                           |             |                             |               |
| `get_device`                | Get appropriate device for computation    | Emulated    | core/device.py              |       ✓       |
| `patch_torch`               | Apply PyTorch patches                     | Native      | core/patch.py               |       ✓       |
| `log_message`               | Log device operations                     | Native      | core/logger.py              |       ✓       |

## Operations (ops)

### Memory Management

| Function Name                | Description                                | Status      | Module Location              | Test Coverage |
|:----------------------------|:------------------------------------------|:------------|:----------------------------|:-------------:|
| `empty_cache`               | Release unoccupied cached memory          | Emulated    | ops/memory/management.py    |       ✓       |
| `memory_stats`              | Return memory allocator statistics        | Emulated    | ops/memory/management.py    |       ✓       |
| `memory_allocated`          | Get current memory occupied by tensors    | Emulated    | ops/memory/management.py    |       ✓       |
| `memory_reserved`           | Get current memory managed by allocator   | Emulated    | ops/memory/management.py    |       ✓       |

### Neural Network Operations

| Function Name                | Description                                | Status      | Module Location              | Test Coverage |
|:----------------------------|:------------------------------------------|:------------|:----------------------------|:-------------:|
| `Linear`                    | Linear layer operations                    | Emulated    | ops/nn/layers.py            |       ✓       |
| `LayerNorm`                 | Layer normalization                        | Emulated    | ops/nn/normalization.py     |       ✓       |
| `Attention`                 | Attention mechanism                        | Emulated    | ops/nn/attention.py         |       ✓       |
| `init_weights`              | Weight initialization                      | Emulated    | ops/nn/init.py              |       ✓       |

### Random Number Generation

| Function Name                | Description                                | Status      | Module Location              | Test Coverage |
|:----------------------------|:------------------------------------------|:------------|:----------------------------|:-------------:|
| `manual_seed`               | Set RNG seed                              | Emulated    | ops/random/generators.py    |       ✓       |
| `randn`                     | Generate random normal tensors             | Emulated    | ops/random/generators.py    |       ✓       |
| `rand`                      | Generate uniform random tensors            | Emulated    | ops/random/generators.py    |       ✓       |

### Stream Management

| Function Name                | Description                                | Status      | Module Location              | Test Coverage |
|:----------------------------|:------------------------------------------|:------------|:----------------------------|:-------------:|
| `Stream`                    | CUDA/MPS stream handling                   | Emulated    | ops/streams/cuda.py         |       ✓       |
| `current_stream`            | Get current stream                         | Emulated    | ops/streams/cuda.py         |       ✓       |
| `synchronize`               | Synchronize streams                        | Emulated    | ops/streams/synchronize.py  |       ✓       |

### Event Management

| Function Name                | Description                                | Status      | Module Location              | Test Coverage |
|:----------------------------|:------------------------------------------|:------------|:----------------------------|:-------------:|
| `Event`                     | CUDA/MPS event handling                    | Emulated    | ops/events/cuda_events.py   |       ✓       |
| `record_event`              | Record an event                            | Emulated    | ops/events/cuda_events.py   |       ✓       |
| `wait_event`                | Wait for an event                          | Emulated    | ops/events/cuda_events.py   |       ✓       |
| `synchronize_event`         | Synchronize with an event                  | Emulated    | ops/events/synchronize.py   |       ✓       |

### Automatic Differentiation

| Function Name                | Description                                | Status      | Module Location              | Test Coverage |
|:----------------------------|:------------------------------------------|:------------|:----------------------------|:-------------:|
| `Function`                  | Base class for custom autograd functions   | Emulated    | ops/autograd/function.py    |       ✓       |
| `Variable`                  | Variable handling for autograd             | Emulated    | ops/autograd/variable.py    |       ✓       |
| `set_grad_enabled`          | Context manager for gradient calculation   | Emulated    | ops/autograd/grad_mode.py   |       ✓       |

### Optimization

| Function Name                | Description                                | Status      | Module Location              | Test Coverage |
|:----------------------------|:------------------------------------------|:------------|:----------------------------|:-------------:|
| `SGD`                       | Stochastic Gradient Descent optimizer      | Emulated    | ops/optim/optimizer.py      |       ✓       |
| `Adam`                      | Adam optimizer                             | Emulated    | ops/optim/optimizer.py      |       ✓       |
| `StepLR`                    | Step learning rate scheduler               | Emulated    | ops/optim/lr_scheduler.py   |       ✓       |

## Utilities

| Function Name                | Description                                | Status      | Module Location              | Test Coverage |
|:----------------------------|:------------------------------------------|:------------|:----------------------------|:-------------:|
| `compile_model`             | Model compilation utilities                | Emulated    | utils/compile.py            |       ✓       |
| `profile_execution`         | Execution profiling tools                  | Native      | utils/profiling.py          |       ✓       |
| `check_device_type`         | Device type validation                     | Native      | utils/type_utils.py         |       ✓       |
| `get_device_info`           | Device information utilities               | Native      | utils/device_utils.py       |       ✓       |
| `handle_device_error`       | Device error handling                      | Native      | utils/error_handling.py     |       ✓       |

**Status Legend:**
- **Emulated:** Functionality provided for both CUDA and MPS
- **Native:** Uses PyTorch's native implementation
- **Stubbed:** Placeholder implementation
- **Not Implemented:** Feature not yet available

**Test Coverage:**
- ✓: Has comprehensive tests
- ⚠: Partial test coverage
- ✗: No tests yet 