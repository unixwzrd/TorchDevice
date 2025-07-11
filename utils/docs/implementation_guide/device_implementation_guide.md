# Device Translation Implementation Guide
*Generated from PyTorch Function Analysis*

## Critical Functions by Category

### 1. Device Creation Functions (Phase 1 - HIGHEST PRIORITY)

#### torch.DeviceObjType
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.Any
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Special type indicating an unconstrained type....

#### torch.cuda.BFloat16Storage
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.BFloat16Tensor
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.BoolStorage
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.BoolTensor
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.ByteStorage
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.ByteTensor
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.CUDAGraph
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Wrapper around a CUDA graph....

#### torch.cuda.CUDAPluggableAllocator
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: CUDA memory allocator loaded from a so file....

#### torch.cuda.Callable
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Deprecated alias to collections.abc.Callable....

#### torch.cuda.CharStorage
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.CharTensor
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.ComplexDoubleStorage
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.ComplexFloatStorage
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.CudaError
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.DeferredCudaCallError
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.Device
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.DoubleStorage
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.DoubleTensor
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.Event
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Wrapper around a CUDA event....

#### torch.cuda.ExternalStream
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Wrapper around an externally allocated CUDA stream....

#### torch.cuda.FloatStorage
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.FloatTensor
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.HalfStorage
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.HalfTensor
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.IntStorage
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.IntTensor
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.LongStorage
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.LongTensor
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.MemPool
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: MemPool represents a pool of memory in a caching allocator. Currently,...

#### torch.cuda.MemPoolContext
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: MemPoolContext holds the currently active pool and stashes the previous...

#### torch.cuda.Optional
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Optional[X] is equivalent to Union[X, None]....

#### torch.cuda.OutOfMemoryError
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Exception raised when device is out of memory...

#### torch.cuda.ShortStorage
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.ShortTensor
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.Stream
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Wrapper around a CUDA stream....

#### torch.cuda.StreamContext
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Context-manager that selects a given stream....

#### torch.cuda.Union
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Union type; Union[X, Y] means either X or Y....

#### torch.cuda.caching_allocator_alloc
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Perform a memory allocation using the CUDA memory allocator....

#### torch.cuda.caching_allocator_delete
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Delete memory allocated using the CUDA memory allocator....

#### torch.cuda.caching_allocator_enable
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Enable or disable the CUDA memory allocator. On by default....

#### torch.cuda.can_device_access_peer
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Check if peer access between two devices is possible....

#### torch.cuda.cast
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Cast a value to a type....

#### torch.cuda.change_current_allocator
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Change the currently used memory allocator to be the one provided....

#### torch.cuda.check_error
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.classproperty
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.clock_rate
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the clock speed of the GPU SM in MHz (megahertz) over the past sample period as given by `nvi......

#### torch.cuda.cudaStatus
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.cudart
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Retrieves the CUDA runtime API module....

#### torch.cuda.current_blas_handle
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return cublasHandle_t pointer to current cuBLAS handle...

#### torch.cuda.current_device
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the index of a currently selected device....

#### torch.cuda.current_stream
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the currently selected :class:`Stream` for a given device....

#### torch.cuda.default_stream
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the default :class:`Stream` for a given device....

#### torch.cuda.device
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Context-manager that changes the selected device....

#### torch.cuda.device_count
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.device_memory_used
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return used global (device) memory in bytes as given by `nvidia-smi` or `amd-smi`....

#### torch.cuda.device_of
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Context-manager that changes the current device to that of given object....

#### torch.cuda.empty_cache
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Release all unoccupied cached memory currently held by the caching...

#### torch.cuda.get_allocator_backend
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a string describing the active allocator backend as set by...

#### torch.cuda.get_arch_list
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return list CUDA architectures this library was compiled for....

#### torch.cuda.get_device_capability
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Get the cuda capability of a device....

#### torch.cuda.get_device_name
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Get the name of a device....

#### torch.cuda.get_device_properties
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Get the properties of a device....

#### torch.cuda.get_gencode_flags
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return NVCC gencode flags this library was compiled with....

#### torch.cuda.get_per_process_memory_fraction
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Get memory fraction for a process....

#### torch.cuda.get_rng_state
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the random number generator state of the specified GPU as a ByteTensor....

#### torch.cuda.get_rng_state_all
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a list of ByteTensor representing the random number states of all devices....

#### torch.cuda.get_stream_from_external
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a :class:`Stream` from an externally allocated CUDA stream....

#### torch.cuda.get_sync_debug_mode
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return current value of debug mode for cuda synchronizing operations....

#### torch.cuda.graph
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Context-manager that captures CUDA work into a :class:`torch.cuda.CUDAGraph` object for later replay......

#### torch.cuda.graph_pool_handle
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return an opaque token representing the id of a graph memory pool....

#### torch.cuda.host_memory_stats
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a dictionary of CUDA memory allocator statistics for a given device....

#### torch.cuda.host_memory_stats_as_nested_dict
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the result of :func:`~torch.cuda.host_memory_stats` as a nested dictionary....

#### torch.cuda.init
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Initialize PyTorch's CUDA state....

#### torch.cuda.initial_seed
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the current random seed of the current GPU....

#### torch.cuda.ipc_collect
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Force collects GPU memory after it has been released by CUDA IPC....

#### torch.cuda.is_available
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.is_bf16_supported
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a bool indicating if the current CUDA/ROCm device supports dtype bfloat16....

#### torch.cuda.is_current_stream_capturing
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return True if CUDA graph capture is underway on the current CUDA stream, False otherwise....

#### torch.cuda.is_initialized
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return whether PyTorch's CUDA state has been initialized....

#### torch.cuda.is_tf32_supported
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a bool indicating if the current CUDA/ROCm device supports dtype tf32....

#### torch.cuda.list_gpu_processes
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a human-readable printout of the running processes and their GPU memory use for a given devic......

#### torch.cuda.lru_cache
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Least-recently-used cache decorator....

#### torch.cuda.make_graphed_callables
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Accept callables (functions or :class:`nn.Module<torch.nn.Module>`\ s) and returns graphed versions....

#### torch.cuda.manual_seed
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the seed for generating random numbers for the current GPU....

#### torch.cuda.manual_seed_all
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the seed for generating random numbers on all GPUs....

#### torch.cuda.max_memory_allocated
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the maximum GPU memory occupied by tensors in bytes for a given device....

#### torch.cuda.max_memory_cached
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Deprecated; see :func:`~torch.cuda.max_memory_reserved`....

#### torch.cuda.max_memory_reserved
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the maximum GPU memory managed by the caching allocator in bytes for a given device....

#### torch.cuda.mem_get_info
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the global free and total GPU memory for a given device using cudaMemGetInfo....

#### torch.cuda.memory_allocated
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the current GPU memory occupied by tensors in bytes for a given device....

#### torch.cuda.memory_cached
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Deprecated; see :func:`~torch.cuda.memory_reserved`....

#### torch.cuda.memory_reserved
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the current GPU memory managed by the caching allocator in bytes for a given device....

#### torch.cuda.memory_snapshot
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a snapshot of the CUDA memory allocator state across all devices....

#### torch.cuda.memory_stats
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a dictionary of CUDA memory allocator statistics for a given device....

#### torch.cuda.memory_stats_as_nested_dict
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the result of :func:`~torch.cuda.memory_stats` as a nested dictionary....

#### torch.cuda.memory_summary
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a human-readable printout of the current memory allocator statistics for a given device....

#### torch.cuda.memory_usage
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the percent of time over the past sample period during which global (device)...

#### torch.cuda.power_draw
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the average power draw of the GPU sensor in mW (MilliWatts)...

#### torch.cuda.reset_accumulated_host_memory_stats
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Reset the "accumulated" (historical) stats tracked by the host memory allocator....

#### torch.cuda.reset_accumulated_memory_stats
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Reset the "accumulated" (historical) stats tracked by the CUDA memory allocator....

#### torch.cuda.reset_max_memory_allocated
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Reset the starting point in tracking maximum GPU memory occupied by tensors for a given device....

#### torch.cuda.reset_max_memory_cached
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Reset the starting point in tracking maximum GPU memory managed by the caching allocator for a given......

#### torch.cuda.reset_peak_host_memory_stats
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Reset the "peak" stats tracked by the host memory allocator....

#### torch.cuda.reset_peak_memory_stats
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Reset the "peak" stats tracked by the CUDA memory allocator....

#### torch.cuda.seed
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the seed for generating random numbers to a random number for the current GPU....

#### torch.cuda.seed_all
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the seed for generating random numbers to a random number on all GPUs....

#### torch.cuda.set_device
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the current device....

#### torch.cuda.set_per_process_memory_fraction
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set memory fraction for a process....

#### torch.cuda.set_rng_state
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the random number generator state of the specified GPU....

#### torch.cuda.set_rng_state_all
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the random number generator state of all devices....

#### torch.cuda.set_stream
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the current stream.This is a wrapper API to set the stream....

#### torch.cuda.set_sync_debug_mode
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the debug mode for cuda synchronizing operations....

#### torch.cuda.stream
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Wrap around the Context-manager StreamContext that selects a given stream....

#### torch.cuda.synchronize
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Wait for all kernels in all streams on a CUDA device to complete....

#### torch.cuda.temperature
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the average temperature of the GPU sensor in Degrees C (Centigrades)....

#### torch.cuda.use_mem_pool
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: A context manager that routes allocations to a given pool....

#### torch.cuda.utilization
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the percent of time over the past sample period during which one or...

#### torch.device
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.get_default_device
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Gets the default ``torch.Tensor`` to be allocated on ``device``...

#### torch.get_device
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.get_device_module
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.mps.device_count
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Returns the number of available MPS devices....

#### torch.profiler_allow_cudagraph_cupti_lazy_reinit_cuda12
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.set_default_device
- **Category**: unknown
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Sets the default ``torch.Tensor`` to be allocated on ``device``.  This...

### 2. Tensor Creation Functions (Phase 1 - HIGHEST PRIORITY)

#### torch.BFloat16Tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.BoolTensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.ByteTensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.CharTensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.DoubleTensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.FloatTensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.HalfTensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.IntTensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.LongTensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.ShortTensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.Tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.TensorType
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.align_tensors
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.as_tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.broadcast_tensors
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: broadcast_tensors(*tensors) -> List of Tensors...

#### torch.fake_quantize_per_tensor_affine
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.is_tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Returns True if `obj` is a PyTorch tensor....

#### torch.nn.functional.Tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.quantize_per_tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.quantize_per_tensor_dynamic
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.scalar_tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.set_default_tensor_type
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.sparse_bsc_tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: sparse_bsc_tensor(ccol_indices, row_indices, values, size=None, *, dtype=None, device=None, pin_memo......

#### torch.sparse_bsr_tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: sparse_bsr_tensor(crow_indices, col_indices, values, size=None, *, dtype=None, device=None, pin_memo......

#### torch.sparse_compressed_tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: sparse_compressed_tensor(compressed_indices, plain_indices, values, size=None, *, dtype=None, layout......

#### torch.sparse_coo_tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: sparse_coo_tensor(indices, values, size=None, *, dtype=None, device=None, pin_memory=False, requires......

#### torch.sparse_csc_tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: sparse_csc_tensor(ccol_indices, row_indices, values, size=None, *, dtype=None, device=None, pin_memo......

#### torch.sparse_csr_tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: sparse_csr_tensor(crow_indices, col_indices, values, size=None, *, dtype=None, device=None, pin_memo......

#### torch.tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.tensor_split
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.tensordot
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Returns a contraction of a and b over multiple dimensions....

#### torch.mps.Tensor
- **Category**: unknown
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

### 3. Device Management Functions (Phase 2 - MEDIUM PRIORITY)

### 4. Events Functions (Phase 2 - MEDIUM PRIORITY)

#### torch.Event
- **Category**: unknown
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: ...

#### torch.mps.Event
- **Category**: unknown
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: Wrapper around an MPS event....

### 5. Streams Functions (Phase 2 - MEDIUM PRIORITY)

#### torch.Stream
- **Category**: unknown
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: ...

#### torch.StreamObjType
- **Category**: unknown
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: ...

### 6. Neural Network Functions (Phase 3 - LOWER PRIORITY)

#### torch.channel_shuffle
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_affine_grid_generator
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_grid_sampler
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_is_acceptable
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.fake_quantize_per_channel_affine
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.hann_window
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.inner
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.mkldnn_adaptive_avg_pool2d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.native_channel_shuffle
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.DType
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: int([x]) -> integer...

#### torch.nn.functional.Optional
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Optional[X] is equivalent to Union[X, None]....

#### torch.nn.functional.Union
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Union type; Union[X, Y] means either X or Y....

#### torch.nn.functional.adaptive_avg_pool1d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.adaptive_avg_pool2d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a 2D adaptive average pooling over an input signal composed of several input planes....

#### torch.nn.functional.adaptive_avg_pool3d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a 3D adaptive average pooling over an input signal composed of several input planes....

#### torch.nn.functional.affine_grid
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Generate 2D or 3D flow field (sampling grid), given a batch of affine matrices :attr:`theta`....

#### torch.nn.functional.assert_int_or_pair
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.avg_pool1d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.avg_pool2d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.avg_pool3d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.boolean_dispatch
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.channel_shuffle
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.conv_tbc
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.fold
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Combine an array of sliding local blocks into a large containing tensor....

#### torch.nn.functional.grid_sample
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute grid sample....

#### torch.nn.functional.handle_torch_function
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Implement a function with checks for ``__torch_function__`` overrides....

#### torch.nn.functional.hardshrink
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.has_torch_function
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Check for __torch_function__ implementations in the elements of an iterable...

#### torch.nn.functional.has_torch_function_unary
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Special case of `has_torch_function` for single inputs....

#### torch.nn.functional.interpolate
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Down/up samples the input....

#### torch.nn.functional.lp_pool1d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a 1D power-average pooling over an input signal composed of several input planes....

#### torch.nn.functional.lp_pool2d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.lp_pool3d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.native_channel_shuffle
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.one_hot
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.pad
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.pixel_shuffle
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.pixel_unshuffle
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.soft_margin_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the soft margin loss....

#### torch.nn.functional.softshrink
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.threshold
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a threshold to each element of the input Tensor....

#### torch.nn.functional.threshold_
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.unfold
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Extract sliding local blocks from a batched input tensor....

#### torch.nn.functional.upsample
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Upsample input....

#### torch.nn.functional.upsample_nearest
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Upsamples the input, using nearest neighbours' pixel values....

#### torch.q_per_channel_axis
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.q_per_channel_scales
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.q_per_channel_zero_points
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.quantize_per_channel
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.mkldnn_max_pool2d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.mkldnn_max_pool3d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.Callable
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Deprecated alias to collections.abc.Callable....

#### torch.nn.functional.adaptive_max_pool1d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.adaptive_max_pool1d_with_indices
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.adaptive_max_pool2d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: adaptive_max_pool2d(input, output_size, return_indices=False)...

#### torch.nn.functional.adaptive_max_pool2d_with_indices
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: adaptive_max_pool2d(input, output_size, return_indices=False)...

#### torch.nn.functional.adaptive_max_pool3d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.adaptive_max_pool3d_with_indices
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.fractional_max_pool2d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.fractional_max_pool2d_with_indices
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.fractional_max_pool3d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.fractional_max_pool3d_with_indices
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.gumbel_softmax
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.has_torch_function_variadic
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Special case of `has_torch_function` that skips tuple creation....

#### torch.nn.functional.max_pool1d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.max_pool1d_with_indices
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.max_pool2d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.max_pool2d_with_indices
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.max_pool3d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.max_pool3d_with_indices
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.max_unpool1d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute a partial inverse of :class:`MaxPool1d`....

#### torch.nn.functional.max_unpool2d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute a partial inverse of :class:`MaxPool2d`....

#### torch.nn.functional.max_unpool3d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute a partial inverse of :class:`MaxPool3d`....

#### torch.nn.functional.softmax
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a softmax function....

#### torch.nn.functional.softmin
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a softmin function....

#### torch.cudnn_convolution_transpose
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.conv_transpose1d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.conv_transpose2d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.conv_transpose3d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_batch_norm
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_convolution
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_convolution_add_relu
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_convolution_relu
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.mkldnn_convolution
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.mkldnn_linear_backward_weights
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.batch_norm
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply Batch Normalization for each channel across a batch of data....

#### torch.nn.functional.binary_cross_entropy
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute Binary Cross Entropy between the target and input probabilities....

#### torch.nn.functional.binary_cross_entropy_with_logits
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute Binary Cross Entropy between target and input logits....

#### torch.nn.functional.celu
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: celu(input, alpha=1., inplace=False) -> Tensor...

#### torch.nn.functional.celu_
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.cross_entropy
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the cross entropy loss between input logits and target....

#### torch.nn.functional.elu
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply the Exponential Linear Unit (ELU) function element-wise....

#### torch.nn.functional.elu_
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.gelu
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.glu
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.group_norm
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply Group Normalization for last certain number of dimensions....

#### torch.nn.functional.instance_norm
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply Instance Normalization independently for each channel in every data sample within a batch....

#### torch.nn.functional.kl_div
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the KL Divergence loss....

#### torch.nn.functional.layer_norm
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply Layer Normalization for last certain number of dimensions....

#### torch.nn.functional.leaky_relu
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.leaky_relu_
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.local_response_norm
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply local response normalization over an input signal....

#### torch.nn.functional.multi_head_attention_forward
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Forward method for MultiHeadAttention....

#### torch.nn.functional.multi_margin_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the multi margin loss, with optional weighting....

#### torch.nn.functional.multilabel_margin_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the multilabel margin loss....

#### torch.nn.functional.multilabel_soft_margin_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the multilabel soft margin loss....

#### torch.nn.functional.normalize
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Perform :math:`L_p` normalization of inputs over specified dimension....

#### torch.nn.functional.pairwise_distance
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.pdist
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.prelu
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: prelu(input, weight) -> Tensor...

#### torch.nn.functional.relu
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: relu(input, inplace=False) -> Tensor...

#### torch.nn.functional.relu6
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: relu6(input, inplace=False) -> Tensor...

#### torch.nn.functional.relu_
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.rms_norm
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply Root Mean Square Layer Normalization....

#### torch.nn.functional.rrelu
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Tensor...

#### torch.nn.functional.rrelu_
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.scaled_dot_product_attention
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,...

#### torch.nn.functional.selu
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: selu(input, inplace=False) -> Tensor...

#### torch.nn.functional.selu_
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.silu
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply the Sigmoid Linear Unit (SiLU) function, element-wise....

#### torch.nn.functional.softplus
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.triplet_margin_with_distance_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the triplet margin loss for input tensors using a custom distance function....

#### torch.quantized_rnn_relu_cell
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.rnn_relu
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.rnn_relu_cell
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.cosine_embedding_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the cosine embedding loss....

#### torch.nn.functional.cosine_similarity
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.hardsigmoid
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply the Hardsigmoid function element-wise....

#### torch.nn.functional.hardswish
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply hardswish function, element-wise....

#### torch.nn.functional.hardtanh
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.hardtanh_
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.log_softmax
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a softmax followed by a logarithm....

#### torch.nn.functional.logsigmoid
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.mish
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply the Mish function, element-wise....

#### torch.nn.functional.sigmoid
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: sigmoid(input) -> Tensor...

#### torch.nn.functional.softsign
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: softsign(input) -> Tensor...

#### torch.nn.functional.tanh
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: tanh(input) -> Tensor...

#### torch.nn.functional.tanhshrink
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: tanhshrink(input) -> Tensor...

#### torch.quantized_rnn_tanh_cell
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.rnn_tanh
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.rnn_tanh_cell
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.miopen_rnn
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.mkldnn_rnn_layer
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.alpha_dropout
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply alpha dropout to the input....

#### torch.nn.functional.bilinear
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.conv1d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.conv2d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.conv3d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.dropout
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: During training, randomly zeroes some elements of the input tensor with probability :attr:`p`....

#### torch.nn.functional.dropout1d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Randomly zero out entire channels (a channel is a 1D feature map)....

#### torch.nn.functional.dropout2d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Randomly zero out entire channels (a channel is a 2D feature map)....

#### torch.nn.functional.dropout3d
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Randomly zero out entire channels (a channel is a 3D feature map)....

#### torch.nn.functional.embedding
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Generate a simple lookup table that looks up embeddings in a fixed dictionary and size....

#### torch.nn.functional.embedding_bag
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute sums, means or maxes of `bags` of embeddings....

#### torch.nn.functional.feature_alpha_dropout
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Randomly masks out entire channels (a channel is a feature map)....

#### torch.nn.functional.hinge_embedding_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the hinge embedding loss....

#### torch.nn.functional.linear
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.upsample_bilinear
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Upsamples the input, using bilinear upsampling....

#### torch.nn.functional.ctc_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the Connectionist Temporal Classification loss....

#### torch.nn.functional.gaussian_nll_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the Gaussian negative log likelihood loss....

#### torch.nn.functional.huber_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the Huber loss, with optional weighting....

#### torch.nn.functional.l1_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the L1 loss, with optional weighting....

#### torch.nn.functional.margin_ranking_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the margin ranking loss....

#### torch.nn.functional.mse_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the element-wise mean squared error, with optional weighting....

#### torch.nn.functional.nll_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the negative log likelihood loss....

#### torch.nn.functional.poisson_nll_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the Poisson negative log likelihood loss....

#### torch.nn.functional.smooth_l1_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the Smooth L1 loss....

#### torch.nn.functional.triplet_margin_loss
- **Category**: unknown
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the triplet loss between given input tensors and a margin greater than 0....

### 7. Device-Specific Functions (Phase 3 - LOWER PRIORITY)

#### torch.from_dlpack
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: from_dlpack(ext_tensor) -> Tensor...

#### torch.get_autocast_xla_dtype
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.get_rng_state
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the random number generator state as a `torch.ByteTensor`....

#### torch.is_autocast_xla_enabled
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.manual_seed
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the seed for generating random numbers on all devices. Returns a...

#### torch.meshgrid
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates grids of coordinates specified by the 1D inputs in `attr`:tensors....

#### torch.parse_ir
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: parse_ir(input: str, parse_tensor_constants: bool = False) -> torch::jit::Graph...

#### torch.set_autocast_xla_dtype
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.set_autocast_xla_enabled
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.to_dlpack
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: to_dlpack(tensor) -> PyCapsule...

#### torch.unique
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> tuple[Tensor, Ten......

#### torch.unravel_index
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Converts a tensor of flat indices into a tuple of coordinate tensors that...

#### torch.cartesian_prod
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Do cartesian product of the given sequence of tensors. The behavior is similar to...

#### torch.block_diag
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Create a block diagonal matrix from provided tensors....

#### torch.split
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Splits the tensor into chunks. Each chunk is a view of the original tensor....

#### torch.get_autocast_cpu_dtype
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.is_autocast_cpu_enabled
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.mps.Union
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Union type; Union[X, Y] means either X or Y....

#### torch.mps.compile_shader
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Compiles compute shader from source and allows one to invoke kernels...

#### torch.mps.empty_cache
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Releases all unoccupied cached memory currently held by the caching...

#### torch.mps.get_rng_state
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the random number generator state as a ByteTensor....

#### torch.mps.is_available
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.mps.manual_seed
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the seed for generating random numbers....

#### torch.mps.seed
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the seed for generating random numbers to a random number....

#### torch.mps.set_rng_state
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the random number generator state....

#### torch.mps.synchronize
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Waits for all kernels in all streams on a MPS device to complete....

#### torch.set_autocast_cpu_dtype
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.set_autocast_cpu_enabled
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.chain_matmul
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the matrix product of the :math:`N` 2-D tensors. This product is efficiently computed...

#### torch.einsum
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: einsum(equation, *operands) -> Tensor...

#### torch.norm
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the matrix norm or vector norm of a given tensor....

#### torch.prelu
- **Category**: unknown
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: prelu(input, weight) -> Tensor...

### 8. Memory Functions (Phase 3 - LOWER PRIORITY)

#### torch.OutOfMemoryError
- **Category**: unknown
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Exception raised when device is out of memory...

#### torch.memory_format
- **Category**: unknown
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: ...

#### torch.mps.current_allocated_memory
- **Category**: unknown
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Returns the current GPU memory occupied by tensors in bytes....

#### torch.mps.driver_allocated_memory
- **Category**: unknown
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Returns total GPU memory allocated by Metal driver for the process in bytes....

#### torch.mps.recommended_max_memory
- **Category**: unknown
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Returns recommended max Working set size for GPU memory in bytes....

#### torch.mps.set_per_process_memory_fraction
- **Category**: unknown
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Set memory fraction for limiting process's memory allocation on MPS device....

## Implementation Phases Summary

### Phase 1: Core Infrastructure (Highest Priority)
- Device creation functions
- Tensor creation functions
- **Total**: 158 functions

### Phase 2: Device Management (Medium Priority)
- Device management functions
- Events and streams
- **Total**: 4 functions

### Phase 3: Advanced Features (Lower Priority)
- Neural network functions
- Device-specific operations
- Memory management
- **Total**: 206 functions

## Next Steps

1. **Start with Phase 1** - Implement core device and tensor creation
2. **Move to Phase 2** - Add device management and CUDA compatibility
3. **Complete Phase 3** - Handle advanced features and optimizations
4. **Test thoroughly** - Ensure all functions work correctly
5. **Update status** - Mark functions as implemented in migration plan
