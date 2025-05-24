# Torch Device

TorchDevice is a migration tool that enables users and developers to run code written for one accelerator on another seamlessly. The module intercepts Torch calls and redirects them to the default accelerator available on the host hardware.

# Torch Device Behavior

TorchDevice intercepts all Torch method and function calls and redirects device creation to the accelerator available on the system. This allows CUDA code to run on MPS unmodified, and MPS code to run on CUDA unmodified. When a redirection occurs, a log message is issued indicating that the Torch call has been intercepted, which device it was redirected to, and where the redirection occurred. This diagnostic information helps developers identify and eventually modify calls that might require hardware-specific adjustments. Although this is primarily a migration tool, it also enables users to run code written for a different accelerator without modification, thereby broadening the available code-base for use on their hardware.

## Device Redirection Policy

| **Device Specification** | **Description** | **Accelerator MPS** | **Accelerator CUDA** | **CPU-Only** | **Notes** |
|--------------------------|-----------------|---------------------|----------------------|--------------|-----------|
| **None / Not Provided** | No device explicitly provided | **Force Accelerated:** Tensor created on MPS | **Force Accelerated:** Tensor created on CUDA | **Fall-back:** Tensor created on CPU if no accelerator is available. | The tensor creation wrapper automatically injects the default accelerator device. |
| **"cpu"** or **"cpu:n"** | Explicit request for CPU | **Redirect:** Tensor created on MPS. | **Redirect:** Tensor created on CUDA. | Remains CPU | Assumes many explicit CPU calls are unintentional; forces accelerated computation unless the developer uses the override. |
| **"cpu:-1"** | Explicit CPU override request | **Override:** Tensor is created on CPU, even if MPS is available. | **Override:** Tensor is created on CPU, even if CUDA is available. | **Override:** Tensor is created on CPU. | Special syntax to request a genuine CPU tensor. Once the override is active, all subsequent explicit CPU requests will yield CPU tensors until the override is toggled off. |
| **"mps"** or **"mps:n"** | Explicit request for MPS | **No redirection:** Tensor is created on MPS if available; TorchDevice does not modify it. | **Redirect:** Tensor created on CUDA if MPS is unavailable. | **Fall-Back:** Tensor created on CPU if neither MPS nor CUDA is available. | If the requested accelerator exists, TorchDevice leaves it unchanged; otherwise, redirection occurs based on system availability. |
| **"cuda"** or **"cuda:n"** | Explicit request for CUDA | **Redirect:** Tensor is created on MPS if available. | **No redirection:** Tensor is created on CUDA if available; TorchDevice does not modify it. | **Fall-Back:** Tensor created on CPU if neither MPS nor CUDA is available. | If MPS is available and preferred over CUDA, a CUDA request may be redirected to MPS. Otherwise, the request is honored. |

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
