# Torch Device

TorchDevice is a migration tool that enables users and developers to run code written for one accelerator on another seamlessly. The module intercepts Torch calls and redirects them to the default accelerator available on the host hardware.

# Torch Device Behavior

TorchDevice provides transparent device handling for PyTorch applications, enabling seamless execution across different accelerators. This document details the behavior of each module and component.

## Core Behavior

### Device Management
- Intercepts all PyTorch device-related calls
- Redirects device requests based on available hardware
- Maintains consistent device state across operations
- Provides fallback mechanisms when requested devices are unavailable

### Device Redirection Policy

| **Device Specification** | **Description** | **Accelerator MPS** | **Accelerator CUDA** | **CPU-Only** | **Notes** |
|--------------------------|-----------------|---------------------|----------------------|--------------|-----------|
| **None / Not Provided** | No device explicitly provided | **Force Accelerated:** Tensor created on MPS | **Force Accelerated:** Tensor created on CUDA | **Fall-back:** Tensor created on CPU if no accelerator is available. | The tensor creation wrapper automatically injects the default accelerator device. |
| **"cpu"** or **"cpu:n"** | Explicit request for CPU | **Redirect:** Tensor created on MPS. | **Redirect:** Tensor created on CUDA. | Remains CPU | Assumes many explicit CPU calls are unintentional; forces accelerated computation unless the developer uses the override. |
| **"cpu:-1"** | Explicit CPU override request | **Override:** Tensor is created on CPU, even if MPS is available. | **Override:** Tensor is created on CPU, even if CUDA is available. | **Override:** Tensor is created on CPU. | Special syntax to request a genuine CPU tensor. Once the override is active, all subsequent explicit CPU requests will yield CPU tensors until the override is toggled off. |
| **"mps"** or **"mps:n"** | Explicit request for MPS | **No redirection:** Tensor is created on MPS if available. | **Redirect:** Tensor created on CUDA if MPS is unavailable. | **Fall-Back:** Tensor created on CPU if neither MPS nor CUDA is available. | If the requested accelerator exists, TorchDevice leaves it unchanged; otherwise, redirection occurs based on system availability. |
| **"cuda"** or **"cuda:n"** | Explicit request for CUDA | **Redirect:** Tensor is created on MPS if available. | **No redirection:** Tensor is created on CUDA if available. | **Fall-Back:** Tensor created on CPU if neither MPS nor CUDA is available. | If MPS is available and preferred over CUDA, a CUDA request may be redirected to MPS. Otherwise, the request is honored. |

## Operations Behavior

### Memory Management
- Tracks memory usage across all devices
- Provides unified memory statistics regardless of device
- Implements device-specific memory optimization strategies
- Handles memory pressure situations gracefully

### Neural Network Operations
- Ensures consistent behavior across devices for all nn modules
- Handles device-specific optimizations transparently
- Maintains numerical stability across different hardware
- Provides fallback implementations for unsupported operations

### Stream Management
- Provides unified stream interface for both CUDA and MPS
- Handles stream synchronization across different devices
- Maintains stream ordering and dependencies
- Implements proper stream cleanup and resource management

### Event Management
- Offers consistent event API across CUDA and MPS
- Handles cross-device event synchronization
- Provides timing and profiling capabilities
- Manages event lifecycle and cleanup

### Automatic Differentiation
- Maintains gradient computation across device transitions
- Handles autograd tape operations consistently
- Provides device-aware backward passes
- Manages gradient accumulation across devices

### Optimization
- Implements device-aware optimizer states
- Handles parameter updates efficiently across devices
- Provides consistent learning rate scheduling
- Manages optimizer memory efficiently

## Utility Behavior

### Compilation
- Provides device-aware model compilation
- Handles device-specific optimizations
- Maintains model integrity across devices
- Implements fallback mechanisms for unsupported features

### Profiling
- Offers unified profiling interface across devices
- Tracks device-specific performance metrics
- Provides memory usage analysis
- Implements timing and throughput measurements

### Error Handling
- Provides clear error messages for device-related issues
- Implements graceful fallback mechanisms
- Maintains detailed error context
- Offers debugging assistance

## Environment Variables

### TORCH_ALLOW_REDIRECTS
Controls device redirection behavior:
- `true`: Always enable device redirection
- Other/unset: Use platform-specific defaults

### TORCHDEVICE_LOG_LEVEL
Controls logging verbosity:
- `DEBUG`: Detailed debugging information
- `INFO`: General operational information
- `WARNING`: Warning messages only
- `ERROR`: Error messages only

## Best Practices

1. **Device Management**
   - Let TorchDevice handle device selection automatically
   - Use explicit device override only when necessary
   - Monitor device transitions in logs

2. **Memory Management**
   - Use context managers for controlled memory usage
   - Monitor memory pressure through provided utilities
   - Implement proper cleanup in your code

3. **Stream and Event Usage**
   - Use streams for concurrent operations
   - Implement proper synchronization
   - Clean up resources explicitly

4. **Optimization**
   - Use device-aware optimizers
   - Monitor memory usage during training
   - Implement gradient clipping when needed

5. **Error Handling**
   - Check device compatibility early
   - Implement proper error handling
   - Monitor logs for device-related issues

### Additional Points

- **Default Behavior:**  
  In systems with a supported accelerator (MPS or CUDA), any tensor creation function that omits the device or explicitly specifies `"cpu"` will be redirected to the accelerator by default. This enforces that the accelerator is used for most computations and reduces the risk of inadvertently performing operations on the CPU.

- **User Notification:**  
  All redirections are logged so that developers are informed that the intended device creation has been redirected. This helps identify lines in code that may need to be migrated to different hardware.

- **Explicit CPU Override:**  
  To intentionally create a CPU tensor on an accelerator system, the developer must use the special `"cpu:-1"` notation. This explicitly signals to TorchDevice that the CPU is desired. When the override is active, explicit CPU requests produce CPU tensors. To revert to default accelerated behavior, specifying `"cpu:-1"` again will remove the override.

- **Consistency in Conversion:**  
  The `.to()` methods are expected to enforce similar redirection rules. For example, calling `.to("cpu")` on a tensor will be redirected to the default accelerator unless the CPU override is active. A tensor created (or overridden) on CPU can be converted to the accelerator device accordingly.

- **Implicit Calls:**  
  Any implicit calls to tensor creation functions without an explicit device are directed to the default accelerator, with a corresponding log message indicating that redirection occurred.

- **Device Specification:**  
  The device specification can be either a string or a `torch.device` object. When a string is provided, it can include a device type and an optional device index.

- **Device Index:**  
  The device index is an integer that specifies the device number. The default device index is 0. On CPU and MPS devices, the index is fixed at 0 (except in special cases), while for CUDA devices it corresponds to the GPU number to use.

- **Device Index Handling:**
  For CPU and MPS devices, PyTorch typically sets the device index to `None` unless an explicit index is provided (e.g., 'mps:0'). TorchDevice defaults the index to `0` for consistency and migration-friendliness, but both `None` and `0` are treated equivalently for these device types. This ensures predictable behavior and compatibility, even if future hardware supports multiple devices.

### Examples

- **Implicit Tensor Creation:**  
  ```python
  tensor = torch.randn(2, 3)
  # On an accelerator system, this tensor is created on "mps" or "cuda" by default.
  ```

- **Explicit CPU Request (Redirected):**  
  ```python
  tensor = torch.tensor([1, 2, 3], device="cpu")
  # This tensor is redirected to the default accelerator (e.g., "mps:0" or "cuda:0").
  ```

- **Explicit CPU Override:**  
  ```python
  tensor = torch.tensor([1, 2, 3], device="cpu:-1")
  # This tensor is created on the CPU, honoring the override.
  ```

- **Conversion:**  
  ```python
  # Creating a tensor (implicitly, on the accelerator).
  tensor = torch.randn(2, 3)
  # Converting to CPU using normal syntax would normally be redirected,
  # so to truly move it to CPU, use the override:
  cpu_tensor = tensor.to("cpu:-1")
  ```

- **Explicit Accelerator Request:**  
  ```python
  tensor_mps = torch.tensor([1, 2, 3], device="mps")
  # This tensor is created on MPS if available.
  ```

### Scope of Redirection

The redirection enforced by TorchDevice is global within the current process or session. This means that once the default accelerator is determined (and unless explicitly overridden), all tensor creation and device conversion calls will adhere to these rules.

For functions which are not directly supported by the TorchDevice module, the original function is called and masqueraded as though it was running on the desired hardware.  That is CUDA calls will respond as CUDA calls but the information will be MPS specific.


| Function Type | Masquerade? | Example Implementation |
|---------------------- |:-----------:|---------------------------------------|
| Device queries | Yes | Return MPS info as CUDA |
| Memory queries | Yes | Use psutil or torch.mps |
| Streams/events | Yes | Provide MPS-backed or dummy objects |
| Mathematic operations | Yes | Use MPS-compatible operations |
| CUDA-only features | No | Stub or raise NotImplementedError |

NOTE: WIll need to find  a graceful method of handling CUDA only functions on MPS.

## Environment Variable Override: TORCH_ALLOW_REDIRECTS

TorchDevice supports an environment variable override to control device redirection behavior for maximum compatibility and migration flexibility.

- If the environment variable `TORCH_ALLOW_REDIRECTS` is set to `true` (case-insensitive), device redirection is always enabled, regardless of platform (including macOS).
- If not set or set to any value other than `true`, the default platform-specific behavior applies (e.g., redirection may be disabled on macOS).
- This override is intended for advanced users, testing, or when running CUDA code on unsupported hardware and you want TorchDevice to transparently redirect to the best available device.

**Example usage:**

```bash
export TORCH_ALLOW_REDIRECTS=true
python your_script.py
```

When this variable is set, TorchDevice will always attempt to redirect device requests according to its policy, even on platforms where this would normally be disabled.

### CUDA Feature Simulation

TorchDevice implements a sophisticated CUDA feature simulation layer for MPS devices. When CUDA-specific features are requested, the system provides appropriate responses that maintain API compatibility while using MPS underneath. This allows CUDA code to run without modification, even when using features that don't have direct MPS equivalents.

#### Simulation Categories

| Feature Type | Simulation Approach | Example Implementation | Notes |
|--------------|-------------------|----------------------|--------|
| **Device Properties** | Full Simulation | Return CUDA-like device properties based on MPS capabilities | `get_device_properties()` returns CUDA-compatible structure |
| **Memory Management** | Direct Mapping | Map CUDA memory calls to MPS equivalents | `memory_allocated()` uses MPS memory tracking |
| **Streams/Events** | Behavioral Simulation | Implement CUDA stream/event semantics using MPS primitives | `cudaStreamCreate()` returns MPS-backed stream object |
| **Kernel Launch** | Function Wrapping | Wrap CUDA kernel launches in MPS-compatible operations | Kernel parameters are translated to MPS format |
| **Synchronization** | Direct Implementation | Implement CUDA sync primitives using MPS equivalents | `cudaDeviceSynchronize()` uses MPS synchronization |
| **Error Handling** | Error Mapping | Map CUDA errors to equivalent MPS/system errors | CUDA error codes are translated to meaningful equivalents |

#### Simulation Strategies

1. **Direct Feature Mapping**
   - When MPS has a direct equivalent, map CUDA calls directly
   - Maintain identical behavior and semantics
   - Example: Memory allocation/deallocation

2. **Behavioral Emulation**
   - For features without direct equivalents, emulate behavior
   - Maintain CUDA-like semantics while using MPS underneath
   - Example: Stream synchronization primitives

3. **State Tracking**
   - Track CUDA-specific state internally
   - Return consistent responses based on tracked state
   - Example: Device properties and capabilities

4. **Fallback Mechanisms**
   - Provide graceful fallbacks for unsupported features
   - Return meaningful errors or default behaviors
   - Example: CUDA-specific optimizations

#### Implementation Examples

```python
# Device Property Simulation
def get_device_properties(device):
    if is_mps_device(device):
        return {
            'name': 'MPS (CUDA Compatible)',
            'major': 7,  # Simulated CUDA compute capability
            'minor': 0,
            'total_memory': get_mps_memory(),
            'multi_processor_count': get_mps_core_count(),
            # Other CUDA-compatible properties
        }

# Stream Simulation
class CUDAStreamSimulator:
    def __init__(self):
        self._mps_stream = create_mps_stream()
        self._cuda_flags = {}  # Track CUDA-specific flags
        
    def synchronize(self):
        self._mps_stream.synchronize()
        
    def wait_event(self, event):
        if isinstance(event, CUDAEventSimulator):
            event._mps_event.wait(self._mps_stream)

# Event Simulation
class CUDAEventSimulator:
    def __init__(self, flags=0):
        self._mps_event = create_mps_event()
        self._cuda_flags = flags
        
    def record(self, stream=None):
        mps_stream = stream._mps_stream if stream else None
        self._mps_event.record(mps_stream)
```

#### Simulation Limitations

Some CUDA features cannot be perfectly simulated on MPS. These cases are handled by:

1. **Graceful Degradation**
   - Return simplified but functionally correct behavior
   - Log warnings when exact CUDA semantics cannot be maintained
   - Example: Complex CUDA-specific optimizations

2. **Feature Substitution**
   - Substitute CUDA-specific features with MPS alternatives
   - Maintain functional equivalence where possible
   - Example: Using MPS-specific memory management

3. **Explicit Unsupported Features**
   - Clearly document unsupported features
   - Raise appropriate exceptions with helpful messages
   - Example: CUDA-only hardware features
