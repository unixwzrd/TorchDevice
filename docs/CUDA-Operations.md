# Torch CUDA Specific Operations
| Function | Description |
|------------------------------------------------|------------------------------------------------|
| StreamContext | Context-manager that selects a given stream. | 
| can_device_access_peer | Check if peer access between two devices is possible. | 
| current_blas_handle | Return cublasHandle_t pointer to current cuBLAS handle | 
| current_device | Return the index of a currently selected device. | 
| current_stream | Return the currently selected Stream for a given device. | 
| cudart | Retrieves the CUDA runtime API module. | 
| default_stream | Return the default Stream for a given device. | 
| device | Context-manager that changes the selected device. | 
| device_count | Return the number of GPUs available. | 
| device_memory_used | Return used global (device) memory in bytes as given by nvidia-smi or amd-smi. | 
| device_of | Context-manager that changes the current device to that of given object. | 
| get_arch_list | Return list CUDA architectures this library was compiled for. | 
| get_device_capability | Get the cuda capability of a device. | 
| get_device_name | Get the name of a device. | 
| get_device_properties | Get the properties of a device. | 
| get_gencode_flags | Return NVCC gencode flags this library was compiled with. | 
| get_sync_debug_mode | Return current value of debug mode for cuda synchronizing operations. | 
| init | Initialize PyTorch's CUDA state. | 
| ipc_collect | Force collects GPU memory after it has been released by CUDA IPC. | 
| is_available | Return a bool indicating if CUDA is currently available. | 
| is_initialized | Return whether PyTorch's CUDA state has been initialized. | 
| memory_usage | Return the percent of time over the past sample period during which global (device) memory was being read or written as given by nvidia-smi. | 
| set_device | Set the current device. | 
| set_stream | Set the current stream.This is a wrapper API to set the stream. | 
| set_sync_debug_mode | Set the debug mode for cuda synchronizing operations. | 
| stream | Wrap around the Context-manager StreamContext that selects a given stream. | 
| synchronize | Wait for all kernels in all streams on a CUDA device to complete. | 
| utilization | Return the percent of time over the past sample period during which one or more kernels was executing on the GPU as given by nvidia-smi. | 
| temperature | Return the average temperature of the GPU sensor in Degrees C (Centigrades). | 
| power_draw | Return the average power draw of the GPU sensor in mW (MilliWatts) | 
| clock_rate | Return the clock speed of the GPU SM in Hz Hertz over the past sample period as given by nvidia-smi. | 
| OutOfMemoryError | Exception raised when device is out of memory | 
|------------------------------------------------|------------------------------------------------|
| **Random Number Generator** | 
| get_rng_state | Return the random number generator state of the specified GPU as a ByteTensor. | 
| get_rng_state_all | Return a list of ByteTensor representing the random number states of all devices. | 
| set_rng_state | Set the random number generator state of the specified GPU. | 
| set_rng_state_all | Set the random number generator state of all devices. | 
| manual_seed | Set the seed for generating random numbers for the current GPU. | 
| manual_seed_all | Set the seed for generating random numbers on all GPUs. | 
| seed | Set the seed for generating random numbers to a random number for the current GPU. | 
| seed_all | Set the seed for generating random numbers to a random number on all GPUs. | 
| initial_seed | Return the current random seed of the current GPU. | 
| Communication collectives | 
| comm.broadcast | Broadcasts a tensor to specified GPU devices. | 
| comm.broadcast_coalesced | Broadcast a sequence of tensors to the specified GPUs. | 
| comm.reduce_add | Sum tensors from multiple GPUs. | 
| comm.scatter | Scatters tensor across multiple GPUs. | 
| comm.gather | Gathers tensors from multiple GPU devices. | 
|------------------------------------------------|------------------------------------------------|
| **Streams and events** | 
| Stream | Wrapper around a CUDA stream. | 
| ExternalStream | Wrapper around an externally allocated CUDA stream. | 
| Event | Wrapper around a CUDA event. | 
|------------------------------------------------|------------------------------------------------|
| **Graphs (beta)** | 
| is_current_stream_capturing | Return True if CUDA graph capture is underway on the current CUDA stream, False otherwise. | 
| graph_pool_handle | Return an opaque token representing the id of a graph memory pool. | 
| CUDAGraph | Wrapper around a CUDA graph. | 
| graph | Context-manager that captures CUDA work into a torch.cuda.CUDAGraph object for later replay. | 
| make_graphed_callables | Accept callables (functions or nn.Modules) and returns graphed versions. | 
|------------------------------------------------|------------------------------------------------|
| **Memory management** | 
| empty_cache | Release all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in nvidia-smi. | 
| get_per_process_memory_fraction | Get memory fraction for a process. | 
| list_gpu_processes | Return a human-readable printout of the running processes and their GPU memory use for a given device. | 
| mem_get_info | Return the global free and total GPU memory for a given device using cudaMemGetInfo. | 
| memory_stats | Return a dictionary of CUDA memory allocator statistics for a given device. | 
| memory_summary | Return a human-readable printout of the current memory allocator statistics for a given device. | 
| memory_snapshot | Return a snapshot of the CUDA memory allocator state across all devices. | 
| memory_allocated | Return the current GPU memory occupied by tensors in bytes for a given device. | 
| max_memory_allocated | Return the maximum GPU memory occupied by tensors in bytes for a given device. | 
| reset_max_memory_allocated | Reset the starting point in tracking maximum GPU memory occupied by tensors for a given device. | 
| memory_reserved | Return the current GPU memory managed by the caching allocator in bytes for a given device. | 
| max_memory_reserved | Return the maximum GPU memory managed by the caching allocator in bytes for a given device. | 
| set_per_process_memory_fraction | Set memory fraction for a process. | 
| memory_cached | Deprecated; see memory_reserved(). | 
| max_memory_cached | Deprecated; see max_memory_reserved(). | 
| reset_max_memory_cached | Reset the starting point in tracking maximum GPU memory managed by the caching allocator for a given device. | 
| reset_peak_memory_stats | Reset the "peak" stats tracked by the CUDA memory allocator. | 
| caching_allocator_alloc | Perform a memory allocation using the CUDA memory allocator. | 
| caching_allocator_delete | Delete memory allocated using the CUDA memory allocator. | 
| get_allocator_backend | Return a string describing the active allocator backend as set by PYTORCH_CUDA_ALLOC_CONF. | 
| CUDAPluggableAllocator | CUDA memory allocator loaded from a so file. | 
| change_current_allocator | Change the currently used memory allocator to be the one provided. | 
| MemPool | MemPool represents a pool of memory in a caching allocator. | 
| MemPoolContext | MemPoolContext holds the currently active pool and stashes the previous pool. | 
| caching_allocator_enable | Enable or disable the CUDA memory allocator. | 
| CLASS torch.cuda.use_mem_pool(pool, device=None)[SOURCE][SOURCE] A context manager that routes allocations to a given pool. | 
|     Parameters
|         pool (torch.cuda.MemPool) – a MemPool object to be made active so that allocations route to this pool.
|         device (torch.device or int, optional) – selected device. Uses MemPool on the current device, given by current_device(), if device is None (default).
|------------------------------------------------|------------------------------------------------|
| **NVIDIA Tools Extension (NVTX)** |  |
| nvtx.mark | Describe an instantaneous event that occurred at some point. | 
| nvtx.range_push | Push a range onto a stack of nested range span. | 
| nvtx.range_pop | Pop a range off of a stack of nested range spans. | 
| nvtx.range | Context manager / decorator that pushes an NVTX range at the beginning of its scope, and pops it at the end. | 
|------------------------------------------------|------------------------------------------------|
| **Jiterator (beta)** | 
| jiterator._create_jit_fn | Create a jiterator-generated cuda kernel for an elementwise op. | 
| jiterator._create_multi_output_jit_fn | Create a jiterator-generated cuda kernel for an elementwise op that supports returning one or more outputs. | 
|------------------------------------------------|------------------------------------------------|
| **TunableOp** | 
| 