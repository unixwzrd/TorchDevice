# TorchDevice Patched/Emulated CUDA Functions

This table tracks all CUDA-related functions that TorchDevice patches, emulates, or stubs. Use it to coordinate modularization, avoid duplication, and ensure regression test coverage. Functions are grouped and named to match the official PyTorch CUDA API for maximum compatibility.

| Function Name                    | Description / Purpose                          | Emulated / Stubbed / Native / Not Implemented | Module / Location         | Migrated | Regression Test |
|----------------------------------|------------------------------------------------|-----------------------------------------------|---------------------------|----------|-----------------|
| **Random/Seed**                  |                                                |                                               |                           |          |                 |
| `manual_seed`                    | Set RNG seed for reproducibility               | Emulated                                      | cuda/math.py              |  [x]     |   [ ]           |
| `manual_seed_all`                | Set RNG seed for all devices                   | Emulated                                      | cuda/math.py              |  [x]     |   [ ]           |
| `seed`                           | Set and return a random seed                   | Emulated                                      | cuda/math.py              |  [x]     |   [ ]           |
| `seed_all`                       | Set random seed for all devices                | Emulated                                      | cuda/math.py              |  [x]     |   [ ]           |
| `get_rng_state`                  | Get RNG state for current device               | Emulated                                      | cuda/math.py              |  [x]     |   [ ]           |
| `get_rng_state_all`              | Get RNG state for all devices                  | Emulated                                      | cuda/math.py              |  [x]     |   [ ]           |
| `set_rng_state`                  | Set RNG state for current device               | Emulated                                      | cuda/math.py              |  [x]     |   [ ]           |
| `set_rng_state_all`              | Set RNG state for all devices                  | Emulated                                      | cuda/math.py              |  [x]     |   [ ]           |
| `initial_seed`                   | Return initial seed                            | Emulated                                      | cuda/math.py              |  [x]     |   [ ]           |
| **Tensor Creation**              |                                                |                                               |                           |          |                 |
| `tensor`                         | Tensor creation with device redirection        | Emulated                                      | cuda/random.py            |  [x]     |   [ ]           |    
| `zeros`                          | Tensor creation with device redirection        | Emulated                                      | cuda/random.py            |  [x]     |   [ ]           |
| `ones`                           | Tensor creation with device redirection        | Emulated                                      | cuda/random.py            |  [x]     |   [ ]           |
| `empty`                          | Tensor creation with device redirection        | Emulated                                      | cuda/random.py            |  [x]     |   [ ]           |
| `randn`                          | Random tensor creation                         | Emulated                                      | cuda/random.py            |  [x]     |   [ ]           |
| `rand`                           | Random tensor creation                         | Emulated                                      | cuda/random.py            |  [x]     |   [ ]           |
| `randint`                        | Random integer tensor creation                 | Emulated                                      | cuda/random.py            |  [x]     |   [ ]           |
| `arange`                         | Range tensor creation                          | Emulated                                      | cuda/random.py            |  [x]     |   [ ]           |
| `linspace`                       | Linspace tensor creation                       | Emulated                                      | cuda/random.py            |  [x]     |   [ ]           |
| `logspace`                       | Logspace tensor creation                       | Emulated                                      | cuda/random.py            |  [x]     |   [ ]           |
| **Memory Management**            |                                                |                                               |                           |          |                 |
| `empty_cache`                    | Release all unoccupied cached memory           | Emulated                                      | cuda/memory.py            |  [x]     |   [ ]           |
| `get_per_process_memory_fraction`| Get memory fraction for a process              | Not Implemented                               |                           |          |                 |
| `list_gpu_processes`             | List GPU processes and memory use              | Stubbed                                        | cuda/unassigned.py        |  [x]     |   [ ]           |
| `mem_get_info`                   | Get memory info                                | Emulated                                      | cuda/memory.py            |  [x]     |   [ ]           |
| `memory_stats`                   | Return memory allocator statistics             | Emulated                                      | cuda/memory.py            |  [x]     |   [ ]           |
| `memory_summary`                 | Human-readable memory allocator stats          | Emulated                                      | cuda/memory.py            |  [x]     |   [ ]           |
| `memory_snapshot`                | Snapshot of memory allocator state             | Emulated                                      | cuda/memory.py            |  [x]     |   [ ]           |
| `memory_allocated`               | Current memory occupied by tensors             | Emulated                                      | cuda/memory.py            |  [x]     |   [ ]           |
| `max_memory_allocated`           | Maximum memory occupied by tensors             | Not Implemented                               |                           |          |                 |
| `reset_max_memory_allocated`     | Reset max memory allocated tracking            | Stubbed                                        | cuda/memory.py            |  [x]     |   [ ]           |
| `memory_reserved`                | Current memory managed by allocator            | Emulated                                      | cuda/memory.py            |  [x]     |   [ ]           |
| `max_memory_reserved`            | Maximum memory managed by allocator            | Not Implemented                               |                           |          |                 |
| `reset_max_memory_reserved`      | Reset max memory reserved tracking             | Stubbed                                        | cuda/memory.py            |  [x]     |   [ ]           |
| `set_per_process_memory_fraction`| Set memory fraction for a process              | Not Implemented                               |                           |          |                 |
| `memory_cached`                  | Deprecated; see memory_reserved                | Not Implemented                               |                           |          |                 |
| `max_memory_cached`              | Deprecated; see max_memory_reserved            | Not Implemented                               |                           |          |                 |
| `reset_max_memory_cached`        | Reset max memory cached tracking               | Not Implemented                               |                           |          |                 |
| `reset_peak_memory_stats`        | Reset "peak" stats tracked by allocator        | Stubbed                                        | cuda/memory.py            |  [x]     |   [ ]           |
| `caching_allocator_alloc`        | Allocate memory using CUDA allocator           | Not Implemented                               |                           |          |                 |
| `caching_allocator_delete`       | Delete memory allocated using CUDA allocator   | Not Implemented                               |                           |          |                 |
| `get_allocator_backend`          | Get active allocator backend                   | Not Implemented                               |                           |          |                 |
| `CUDAPluggableAllocator`         | CUDA memory allocator from .so file            | Not Implemented                               |                           |          |                 |
| `change_current_allocator`       | Change the current memory allocator            | Not Implemented                               |                           |          |                 |
| `MemPool`                        | Pool of memory in caching allocator            | Not Implemented                               |                           |          |                 |
| `MemPoolContext`                 | Context for active memory pool                 | Not Implemented                               |                           |          |                 |
| `caching_allocator_enable`       | Enable/disable CUDA memory allocator           | Not Implemented                               |                           |          |                 |
| `use_mem_pool`                   | Context manager for memory pool allocations    | Not Implemented                               |                           |          |                 |
| **Streams/Events**               |                                                |                                               |                           |          |                 |
| `Stream`                         | CUDA stream (MPS/CPU functional)               | Emulated                                      | cuda/streams.py           |  [x]     |   [ ]           |
| `ExternalStream`                 | Externally allocated CUDA stream               | Not Implemented                               |                           |          |                 |
| `Event`                          | CUDA event (MPS/CPU functional)                | Emulated                                      | cuda/streams.py           |  [x]     |   [ ]           |
| **Graphs (beta)**                |                                                |                                               |                           |          |                 |
| `is_current_stream_capturing`    | Return True if CUDA graph capture is underway  | Stubbed                                        | cuda/unassigned.py        |  [x]     |   [ ]           |
| `graph_pool_handle`              | Opaque token for graph memory pool             | Stubbed                                        | cuda/unassigned.py        |  [x]     |   [ ]           |
| `CUDAGraph`                      | CUDA graph class                               | Stubbed                                        | cuda/unassigned.py        |  [x]     |   [ ]           |
| `graph`                          | Context-manager for CUDA graph capture         | Stubbed                                        | cuda/unassigned.py        |  [x]     |   [ ]           |
| `make_graphed_callables`         | Return graphed versions of callables           | Stubbed                                        | cuda/unassigned.py        |  [x]     |   [ ]           |
| **Device/Properties**            |                                                |                                               |                           |          |                 |
| `set_device`                     | Set current device                             | Emulated                                      | cuda/device.py            |  [x]     |   [ ]           |
| `current_device`                 | Get current device index                       | Emulated                                      | cuda/device.py            |  [x]     |   [ ]           |
| `device_count`                   | Get number of devices                          | Emulated                                      | cuda/device.py            |  [x]     |   [ ]           |
| `is_available`                   | Check if CUDA/MPS is available                 | Emulated                                      | cuda/device.py            |  [x]     |   [ ]           |
| `get_device_name`                | Get device name                                | Emulated                                      | cuda/device.py            |  [x]     |   [ ]           |
| `get_device_capability`          | Get device capability                          | Emulated                                      | cuda/device.py            |  [x]     |   [ ]           |
| `is_initialized`                 | Check if CUDA/MPS is initialized               | Emulated                                      | cuda/device.py            |  [x]     |   [ ]           |
| `get_arch_list`                  | Get architecture list                          | Emulated                                      | cuda/device.py            |  [x]     |   [ ]           |
| `is_built`                       | Check if CUDA/MPS is built                     | Emulated                                      | cuda/device.py            |  [x]     |   [ ]           |
| `device`                         | Context-manager for device selection           | Emulated                                      | cuda/device.py            |  [x]     |   [ ]           |
| `device_memory_used`             | Used global (device) memory in bytes           | Not Implemented                               |                           |          |                 |
| `device_of`                      | Context-manager for device of given object     | Not Implemented                               |                           |          |                 |
| `ipc_collect`                    | Force collect GPU memory after IPC             | Not Implemented                               |                           |          |                 |
| **Other/Unsupported**            |                                                |                                               |                           |          |                 |
| `jiterator`                      | CUDA JIT iterator (unsupported)                | Stubbed                                        | cuda/unassigned.py        |  [x]     |   [ ]           |
| `comm.broadcast`                 | Broadcast tensor to specified devices          | Not Implemented                               |                           |          |                 |
| `comm.broadcast_coalesced`       | Broadcast sequence of tensors                  | Not Implemented                               |                           |          |                 |
| `comm.reduce_add`                | Sum tensors from multiple devices              | Not Implemented                               |                           |          |                 |
| `comm.scatter`                   | Scatter tensor across devices                  | Not Implemented                               |                           |          |                 |
| `comm.gather`                    | Gather tensors from multiple devices           | Not Implemented                               |                           |          |                 |
| `nvtx.mark`                      | NVTX event marker                              | Not Implemented                               |                           |          |                 |
| `nvtx.range_push`                | Push NVTX range                                | Not Implemented                               |                           |          |                 |
| `nvtx.range_pop`                 | Pop NVTX range                                 | Not Implemented                               |                           |          |                 |
| `nvtx.range`                     | NVTX range context manager/decorator           | Not Implemented                               |                           |          |                 |


**Legend:**
- **Emulated:** Functionality is provided for MPS/CPU, not just stubbed.
- **Stubbed:** No-op or placeholder for unsupported CUDA features.
- **Not Implemented:** Not yet present in TorchDevice.
- **Native:** Uses PyTorch's native implementation (rare in this context).
- **Migrated:** [x] if patching is now in a cuda/* module, blank if not.
- **Regression Test:** Mark when a regression test exists for this function.

Add more functions as you modularize or discover them. Group related functions together for easier tracking and migration. 