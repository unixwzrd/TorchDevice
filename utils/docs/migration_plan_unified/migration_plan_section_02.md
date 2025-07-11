# TorchDevice Migration Plan
*Generated from PyTorch Function Analysis*

## Overview
This document contains the complete migration plan for implementing TorchDevice, 
a PyTorch device translation layer that enables seamless switching between 
CUDA, MPS, and CPU backends.

## Migration Strategy

### Priority Matrix
- **🔴 Critical**: Core device functions (torch.device, torch.tensor, etc.)
- **🟡 High**: Neural network operations, optimization functions
- **🟢 Medium**: Mathematical operations, utilities
- **🔵 Low**: Specialized operations, experimental features

### Implementation Phases
1. **Phase 1**: Core device management and tensor creation
2. **Phase 2**: Neural network operations and optimization
3. **Phase 3**: Mathematical operations and utilities
4. **Phase 4**: Specialized operations and edge cases

### Architecture Overview
- Device translation layer intercepts PyTorch calls
- Automatic fallback from CUDA → MPS → CPU
- Transparent API compatibility
- Performance monitoring and optimization

### Risk Mitigation
- Comprehensive testing across device combinations
- Gradual rollout with feature flags
- Performance benchmarking and optimization
- Backward compatibility maintenance

### Success Metrics
- 100% API compatibility with PyTorch
- <5% performance overhead
- Zero breaking changes for existing code
- Successful migration of test projects

## Function Catalog

| Category | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
|:---------|:---:|:----:|:---:|:---:|:------:|-----------|-------------|-------|

| 🟦 CUDA_OPERATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
|:-----------------|:---:|:----:|:---:|:---:|:------:|-----------|-------------|-------|
| `torch.cuda.caching_allocator_alloc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size, device, stream` | `Any` | Perform a memory allocation using the CUDA memory allocator. Memory is allocated for a given device and a stream, this function is intended to be used for interoperability with other frameworks. Al... |
| `torch.cuda.caching_allocator_delete` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mem_ptr` | `Any` | Delete memory allocated using the CUDA memory allocator. Memory allocated with :func:`~torch.cuda.caching_allocator_alloc`. is freed here. The associated device and stream are tracked inside the al... |
| `torch.cuda.caching_allocator_enable` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `None` | Enable or disable the CUDA memory allocator. On by default. |
| `torch.cuda.can_device_access_peer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device, peer_device` | `<class 'bool'>` | Check if peer access between two devices is possible. |
| `torch.cuda.cast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typ, val` | `Any` | Cast a value to a type. This returns the value unchanged. To the type checker this signals that the return value has the designated type, but at runtime we intentionally don't check anything (we wa... |
| `torch.cuda.change_current_allocator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `allocator` | `None` | Change the currently used memory allocator to be the one provided. If the current allocator has already been used/initialized, this function will error. Args: allocator (torch.cuda.memory._CUDAAllo... |
| `torch.cuda.check_error` | ❓ | ❓ | ❓ | ❓ | 🔴 | `res` | `None` |  |
| `torch.cuda.classproperty` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` |  |
| `torch.cuda.clock_rate` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the clock speed of the GPU SM in MHz (megahertz) over the past sample period as given by `nvidia-smi`. Args: device (torch.device or int, optional): selected device. Returns statistic for th... |
| `torch.cuda.cudart` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Retrieves the CUDA runtime API module. This function initializes the CUDA runtime environment if it is not already initialized and returns the CUDA runtime API module (_cudart). The CUDA runtime AP... |
| `torch.cuda.current_blas_handle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return cublasHandle_t pointer to current cuBLAS handle |
| `torch.cuda.current_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the index of a currently selected device. |
| `torch.cuda.current_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.cuda.streams.Stream'>` | Return the currently selected :class:`Stream` for a given device. Args: device (torch.device or int, optional): selected device. Returns the currently selected :class:`Stream` for the current devic... |
| `torch.cuda.default_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.cuda.streams.Stream'>` | Return the default :class:`Stream` for a given device. Args: device (torch.device or int, optional): selected device. Returns the default :class:`Stream` for the current device, given by :func:`~to... |
| `torch.cuda.device_count` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the number of GPUs available. .. note:: This API will NOT posion fork if NVML discovery succeeds. See :ref:`multiprocessing-poison-fork-note` for more details. |
| `torch.cuda.device_memory_used` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return used global (device) memory in bytes as given by `nvidia-smi` or `amd-smi`. Args: device (torch.device or int, optional): selected device. Returns statistic for the current device, given by ... |
| `torch.cuda.empty_cache` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Release all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in `nvidia-smi`. .. note:: :func:`~torch.cuda.empty_cache... |
| `torch.cuda.get_allocator_backend` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'str'>` | Return a string describing the active allocator backend as set by ``PYTORCH_CUDA_ALLOC_CONF``. Currently available backends are ``native`` (PyTorch's native caching allocator) and `cudaMallocAsync`... |
| `torch.cuda.get_arch_list` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `list[str]` | Return list CUDA architectures this library was compiled for. |
| `torch.cuda.get_device_capability` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `tuple[int, int]` | Get the cuda capability of a device. Args: device (torch.device or int or str, optional): device for which to return the device capability. This function is a no-op if this argument is a negative i... |
| `torch.cuda.get_device_name` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'str'>` | Get the name of a device. Args: device (torch.device or int or str, optional): device for which to return the name. This function is a no-op if this argument is a negative integer. It uses the curr... |
| `torch.cuda.get_device_properties` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch._utils._CudaDeviceProperties'>` | Get the properties of a device. Args: device (torch.device or int or str, optional): device for which to return the properties of the device. It uses the current device, given by :func:`~torch.cuda... |
| `torch.cuda.get_gencode_flags` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'str'>` | Return NVCC gencode flags this library was compiled with. |
| `torch.cuda.get_per_process_memory_fraction` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'float'>` | Get memory fraction for a process. Args: device (torch.device or int, optional): selected device. If it is ``None`` the default CUDA device is used. Returns: memory fraction, in range 0~1. Allowed ... |
| `torch.cuda.get_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.Tensor'>` | Return the random number generator state of the specified GPU as a ByteTensor. Args: device (torch.device or int, optional): The device to return the RNG state of. Default: ``'cuda'`` (i.e., ``torc... |
| `torch.cuda.get_rng_state_all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `list[torch.Tensor]` | Return a list of ByteTensor representing the random number states of all devices. |
| `torch.cuda.get_stream_from_external` | ❓ | ❓ | ❓ | ❓ | 🔴 | `data_ptr, device` | `<class 'torch.cuda.streams.Stream'>` | Return a :class:`Stream` from an externally allocated CUDA stream. This function is used to wrap streams allocated in other libraries in order to facilitate data exchange and multi-library interact... |
| `torch.cuda.get_sync_debug_mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return current value of debug mode for cuda synchronizing operations. |
| `torch.cuda.graph_pool_handle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return an opaque token representing the id of a graph memory pool. See :ref:`Graph memory management<graph-memory-management>`. .. warning:: This API is in beta and may change in future releases. |
| `torch.cuda.host_memory_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `dict[str, typing.Any]` | Return a dictionary of CUDA memory allocator statistics for a given device. The return value of this function is a dictionary of statistics, each of which is a non-negative integer. Core statistics... |
| `torch.cuda.host_memory_stats_as_nested_dict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `dict[str, typing.Any]` | Return the result of :func:`~torch.cuda.host_memory_stats` as a nested dictionary. |
| `torch.cuda.init` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Initialize PyTorch's CUDA state. You may need to call this explicitly if you are interacting with PyTorch via its C API, as Python bindings for CUDA functionality will not be available until this i... |
| `torch.cuda.initial_seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the current random seed of the current GPU. .. warning:: This function eagerly initializes CUDA. |
| `torch.cuda.ipc_collect` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Force collects GPU memory after it has been released by CUDA IPC. .. note:: Checks if any sent CUDA tensors could be cleaned from the memory. Force closes shared memory file used for reference coun... |
| `torch.cuda.is_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Return a bool indicating if CUDA is currently available. .. note:: This function will NOT poison fork if the environment variable ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set. For more details, see :... |
| `torch.cuda.is_bf16_supported` | ❓ | ❓ | ❓ | ❓ | 🔴 | `including_emulation` | `Any` | Return a bool indicating if the current CUDA/ROCm device supports dtype bfloat16. |
| `torch.cuda.is_current_stream_capturing` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return True if CUDA graph capture is underway on the current CUDA stream, False otherwise. If a CUDA context does not exist on the current device, returns False without initializing the context. |
| `torch.cuda.is_initialized` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return whether PyTorch's CUDA state has been initialized. |
| `torch.cuda.is_tf32_supported` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Return a bool indicating if the current CUDA/ROCm device supports dtype tf32. |
| `torch.cuda.list_gpu_processes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'str'>` | Return a human-readable printout of the running processes and their GPU memory use for a given device. This can be useful to display periodically during training, or when handling out-of-memory exc... |
| `torch.cuda.lru_cache` | ❓ | ❓ | ❓ | ❓ | 🔴 | `maxsize, typed` | `Any` | Least-recently-used cache decorator. If *maxsize* is set to None, the LRU features are disabled and the cache can grow without bound. If *typed* is True, arguments of different types will be cached... |
| `torch.cuda.make_graphed_callables` | ❓ | ❓ | ❓ | ❓ | 🔴 | `callables, sample_args, num_warmup_iters, ...` | `Any` | Accept callables (functions or :class:`nn.Module<torch.nn.Module>`\ s) and returns graphed versions. Each graphed callable's forward pass runs its source callable's forward CUDA work as a CUDA grap... |
| `torch.cuda.manual_seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `seed` | `None` | Set the seed for generating random numbers for the current GPU. It's safe to call this function if CUDA is not available; in that case, it is silently ignored. Args: seed (int): The desired seed. .... |
| `torch.cuda.manual_seed_all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `seed` | `None` | Set the seed for generating random numbers on all GPUs. It's safe to call this function if CUDA is not available; in that case, it is silently ignored. Args: seed (int): The desired seed. |
| `torch.cuda.max_memory_allocated` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the maximum GPU memory occupied by tensors in bytes for a given device. By default, this returns the peak allocated memory since the beginning of this program. :func:`~torch.cuda.reset_peak_... |
| `torch.cuda.max_memory_cached` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Deprecated; see :func:`~torch.cuda.max_memory_reserved`. |
| `torch.cuda.max_memory_reserved` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the maximum GPU memory managed by the caching allocator in bytes for a given device. By default, this returns the peak cached memory since the beginning of this program. :func:`~torch.cuda.r... |
| `torch.cuda.mem_get_info` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `tuple[int, int]` | Return the global free and total GPU memory for a given device using cudaMemGetInfo. Args: device (torch.device or int or str, optional): selected device. Returns statistic for the current device, ... |
| `torch.cuda.memory_allocated` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the current GPU memory occupied by tensors in bytes for a given device. Args: device (torch.device or int, optional): selected device. Returns statistic for the current device, given by :fun... |
| `torch.cuda.memory_cached` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Deprecated; see :func:`~torch.cuda.memory_reserved`. |
| `torch.cuda.memory_reserved` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the current GPU memory managed by the caching allocator in bytes for a given device. Args: device (torch.device or int, optional): selected device. Returns statistic for the current device, ... |
| `torch.cuda.memory_snapshot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return a snapshot of the CUDA memory allocator state across all devices. Interpreting the output of this function requires familiarity with the memory allocator internals. .. note:: See :ref:`cuda-... |
| `torch.cuda.memory_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `dict[str, typing.Any]` | Return a dictionary of CUDA memory allocator statistics for a given device. The return value of this function is a dictionary of statistics, each of which is a non-negative integer. Core statistics... |
| `torch.cuda.memory_stats_as_nested_dict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `dict[str, typing.Any]` | Return the result of :func:`~torch.cuda.memory_stats` as a nested dictionary. |
| `torch.cuda.memory_summary` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device, abbreviated` | `<class 'str'>` | Return a human-readable printout of the current memory allocator statistics for a given device. This can be useful to display periodically during training, or when handling out-of-memory exceptions... |
| `torch.cuda.memory_usage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the percent of time over the past sample period during which global (device) memory was being read or written as given by `nvidia-smi`. Args: device (torch.device or int, optional): selected... |
| `torch.cuda.power_draw` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the average power draw of the GPU sensor in mW (MilliWatts) over the past sample period as given by `nvidia-smi` for Fermi or newer fully supported devices. Args: device (torch.device or int... |
| `torch.cuda.reset_accumulated_host_memory_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Reset the "accumulated" (historical) stats tracked by the host memory allocator. See :func:`~torch.cuda.host_memory_stats` for details. Accumulated stats correspond to the `"allocated"` and `"freed... |
| `torch.cuda.reset_accumulated_memory_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Reset the "accumulated" (historical) stats tracked by the CUDA memory allocator. See :func:`~torch.cuda.memory_stats` for details. Accumulated stats correspond to the `"allocated"` and `"freed"` ke... |
| `torch.cuda.reset_max_memory_allocated` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Reset the starting point in tracking maximum GPU memory occupied by tensors for a given device. See :func:`~torch.cuda.max_memory_allocated` for details. Args: device (torch.device or int, optional... |
| `torch.cuda.reset_max_memory_cached` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Reset the starting point in tracking maximum GPU memory managed by the caching allocator for a given device. See :func:`~torch.cuda.max_memory_cached` for details. Args: device (torch.device or int... |
| `torch.cuda.reset_peak_host_memory_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Reset the "peak" stats tracked by the host memory allocator. See :func:`~torch.cuda.host_memory_stats` for details. Peak stats correspond to the `"peak"` key in each individual stat dict. |
| `torch.cuda.reset_peak_memory_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Reset the "peak" stats tracked by the CUDA memory allocator. See :func:`~torch.cuda.memory_stats` for details. Peak stats correspond to the `"peak"` key in each individual stat dict. Args: device (... |
| `torch.cuda.seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Set the seed for generating random numbers to a random number for the current GPU. It's safe to call this function if CUDA is not available; in that case, it is silently ignored. .. warning:: If yo... |
| `torch.cuda.seed_all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Set the seed for generating random numbers to a random number on all GPUs. It's safe to call this function if CUDA is not available; in that case, it is silently ignored. |
| `torch.cuda.set_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Set the current device. Usage of this function is discouraged in favor of :any:`device`. In most cases it's better to use ``CUDA_VISIBLE_DEVICES`` environmental variable. Args: device (torch.device... |
| `torch.cuda.set_per_process_memory_fraction` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fraction, device` | `None` | Set memory fraction for a process. The fraction is used to limit an caching allocator to allocated memory on a CUDA device. The allowed value equals the total visible memory multiplied fraction. If... |
| `torch.cuda.set_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `new_state, device` | `None` | Set the random number generator state of the specified GPU. Args: new_state (torch.ByteTensor): The desired state device (torch.device or int, optional): The device to set the RNG state. Default: `... |
| `torch.cuda.set_rng_state_all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `new_states` | `None` | Set the random number generator state of all devices. Args: new_states (Iterable of torch.ByteTensor): The desired state for each device. |
| `torch.cuda.set_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `Any` | Set the current stream.This is a wrapper API to set the stream. Usage of this function is discouraged in favor of the ``stream`` context manager. Args: stream (Stream): selected stream. This functi... |
| `torch.cuda.set_sync_debug_mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `debug_mode` | `None` | Set the debug mode for cuda synchronizing operations. Args: debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations, if "warn" or 1, warn on synchronizing operati... |
| `torch.cuda.stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `<class 'torch.cuda.StreamContext'>` | Wrap around the Context-manager StreamContext that selects a given stream. Arguments: stream (Stream): selected stream. This manager is a no-op if it's ``None``. .. note:: In eager mode stream is o... |
| `torch.cuda.synchronize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Wait for all kernels in all streams on a CUDA device to complete. Args: device (torch.device or int, optional): device for which to synchronize. It uses the current device, given by :func:`~torch.c... |
| `torch.cuda.temperature` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the average temperature of the GPU sensor in Degrees C (Centigrades). The average temperature is computed based on past sample period as given by `nvidia-smi`. Args: device (torch.device or ... |
| `torch.cuda.use_mem_pool` | ❓ | ❓ | ❓ | ❓ | 🔴 | `pool, device` | `Any` | A context manager that routes allocations to a given pool. Args: pool(torch.cuda.MemPool): a MemPool object to be made active so that allocations route to this pool. device (torch.device or int, op... |
| `torch.cuda.utilization` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the percent of time over the past sample period during which one or more kernels was executing on the GPU as given by `nvidia-smi`. Args: device (torch.device or int, optional): selected dev... |
| | | | | | | | | |
| 🟦 DISTRIBUTED_COMPUTING | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.distributed.all_gather` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor_list, tensor, group, ...` | `Any` | Gathers tensors from the whole group in a list. Complex and uneven sized tensors are supported. Args: tensor_list (list[Tensor]): Output list. It should contain correctly-sized tensors to be used f... |
| `torch.distributed.all_gather_coalesced` | ❓ | ❓ | ❓ | ❓ | 🔴 | `output_tensor_lists, input_tensor_list, group, ...` | `Any` | Gathers input tensors from the whole group in a list in a coalesced manner. Complex tensors are supported. Args: output_tensor_lists (list[list[Tensor]]): Output list. It should contain correctly-s... |
| `torch.distributed.all_gather_into_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `output_tensor, input_tensor, group, ...` | `Any` | Gather tensors from all ranks and put them in a single output tensor. This function requires all tensors to be the same size on each process. Args: output_tensor (Tensor): Output tensor to accommod... |
| `torch.distributed.all_gather_object` | ❓ | ❓ | ❓ | ❓ | 🔴 | `object_list, obj, group` | `Any` | Gathers picklable objects from the whole group into a list. Similar to :func:`all_gather`, but Python objects can be passed in. Note that the object must be picklable in order to be gathered. Args:... |
| `torch.distributed.all_reduce` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, op, group, ...` | `Any` | Reduces the tensor data across all machines in a way that all get the final result. After the call ``tensor`` is going to be bitwise identical in all processes. Complex tensors are supported. Args:... |
| `torch.distributed.all_reduce_coalesced` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, op, group, ...` | `Any` | WARNING: at this time individual shape checking is not implemented across nodes. For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the rank 1 node passes [torch.rand(2), tor... |
| `torch.distributed.all_to_all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `output_tensor_list, input_tensor_list, group, ...` | `Any` | Scatters list of input tensors to all processes in a group and return gathered list of tensors in output list. Complex tensors are supported. Args: output_tensor_list (list[Tensor]): List of tensor... |
| `torch.distributed.all_to_all_single` | ❓ | ❓ | ❓ | ❓ | 🔴 | `output, input, output_split_sizes, ...` | `Any` | Split input tensor and then scatter the split list to all processes in a group. Later the received tensors are concatenated from all the processes in the group and returned as a single output tenso... |
| `torch.distributed.barrier` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group, async_op, device_ids` | `Any` | Synchronize all processes. This collective blocks processes until the whole group enters this function, if async_op is False, or if async work handle is called on wait(). Args: group (ProcessGroup,... |
| `torch.distributed.batch_isend_irecv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `p2p_op_list` | `list[torch.distributed.distributed_c10d.Work]` | Send or Receive a batch of tensors asynchronously and return a list of requests. Process each of the operations in ``p2p_op_list`` and return the corresponding requests. NCCL, Gloo, and UCC backend... |
| `torch.distributed.breakpoint` | ❓ | ❓ | ❓ | ❓ | 🔴 | `rank, skip` | `Any` | Set a breakpoint, but only on a single rank. All other ranks will wait for you to be done with the breakpoint before continuing. Args: rank (int): Which rank to break on. Default: ``0`` skip (int):... |
| `torch.distributed.broadcast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, src, group, ...` | `Any` | Broadcasts the tensor to the whole group. ``tensor`` must have the same number of elements in all processes participating in the collective. Args: tensor (Tensor): Data to be sent if ``src`` is the... |
| `torch.distributed.broadcast_object_list` | ❓ | ❓ | ❓ | ❓ | 🔴 | `object_list, src, group, ...` | `Any` | Broadcasts picklable objects in ``object_list`` to the whole group. Similar to :func:`broadcast`, but Python objects can be passed in. Note that all objects in ``object_list`` must be picklable in ... |
| `torch.distributed.destroy_process_group` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group` | `Any` | Destroy a given process group, and deinitialize the distributed package. Args: group (ProcessGroup, optional): The process group to be destroyed, if group.WORLD is given, all process groups includi... |
| `torch.distributed.gather` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, gather_list, dst, ...` | `Any` | Gathers a list of tensors in a single process. This function requires all tensors to be the same size on each process. Args: tensor (Tensor): Input tensor. gather_list (list[Tensor], optional): Lis... |
| `torch.distributed.gather_object` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj, object_gather_list, dst, ...` | `Any` | Gathers picklable objects from the whole group in a single process. Similar to :func:`gather`, but Python objects can be passed in. Note that the object must be picklable in order to be gathered. A... |
| `torch.distributed.get_backend` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group` | `<class 'torch.distributed.distributed_c10d.Backend'>` | Return the backend of the given process group. Args: group (ProcessGroup, optional): The process group to work on. The default is the general main process group. If another specific group is specif... |
| `torch.distributed.get_backend_config` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group` | `<class 'str'>` | Return the backend configuration of the given process group. Args: group (ProcessGroup, optional): The process group to work on. The default is the general main process group. If another specific g... |
| `torch.distributed.get_default_backend_for_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'str'>` | Return the default backend for the given device. Args: Union[str, torch.device]: The device to get the default backend for. Returns: The default backend for the given device as a lower case string. |
| `torch.distributed.get_global_rank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group, group_rank` | `<class 'int'>` | Translate a group rank into a global rank. ``group_rank`` must be part of `group` otherwise this raises RuntimeError. Args: group (ProcessGroup): ProcessGroup to find the global rank from. group_ra... |
| `torch.distributed.get_group_rank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group, global_rank` | `<class 'int'>` | Translate a global rank into a group rank. ``global_rank`` must be part of ``group`` otherwise this raises RuntimeError. Args: group (ProcessGroup): ProcessGroup to find the relative rank. global_r... |
| `torch.distributed.get_node_local_rank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fallback_rank` | `<class 'int'>` | Return the local rank of the current process relative to the node. Semantically, this is a useful concept for mapping processes to devices. For example, on a node with 8 accelerator you could use t... |
| `torch.distributed.get_pg_count` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the number of process groups. |
| `torch.distributed.get_process_group_ranks` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group` | `list[int]` | Get all ranks associated with ``group``. Args: group (ProcessGroup): ProcessGroup to get all ranks from. Returns: List of global ranks ordered by group rank. |
| `torch.distributed.get_rank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group` | `<class 'int'>` | Return the rank of the current process in the provided ``group``, default otherwise. Rank is a unique identifier assigned to each process within a distributed process group. They are always consecu... |
| `torch.distributed.get_world_size` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group` | `<class 'int'>` | Return the number of processes in the current process group. Args: group (ProcessGroup, optional): The process group to work on. If None, the default process group will be used. Returns: The world ... |
| `torch.distributed.init_device_mesh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device_type, mesh_shape, mesh_dim_names` | `<class 'torch.distributed.device_mesh.DeviceMesh'>` | Initializes a `DeviceMesh` based on `device_type`, `mesh_shape`, and `mesh_dim_names` parameters. This creates a DeviceMesh with an n-dimensional array layout, where `n` is the length of `mesh_shap... |
| `torch.distributed.init_process_group` | ❓ | ❓ | ❓ | ❓ | 🔴 | `backend, init_method, timeout, ...` | `None` | Initialize the default distributed process group. This will also initialize the distributed package. There are 2 main ways to initialize a process group: 1. Specify ``store``, ``rank``, and ``world... |
| `torch.distributed.irecv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, src, group, ...` | `typing.Optional[torch.distributed.distributed_c10d.Work]` | Receives a tensor asynchronously. .. warning:: ``tag`` is not supported with the NCCL backend. Unlike recv, which is blocking, irecv allows src == dst rank, i.e. recv from self. Args: tensor (Tenso... |
| `torch.distributed.is_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Return ``True`` if the distributed package is available. Otherwise, ``torch.distributed`` does not expose any other APIs. Currently, ``torch.distributed`` is available on Linux, MacOS and Windows. ... |
| `torch.distributed.is_backend_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `backend` | `<class 'bool'>` | Check backend availability. Checks if the given backend is available and supports the built-in backends or third-party backends through function ``Backend.register_backend``. Args: backend (str): B... |
| `torch.distributed.is_gloo_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Check if the Gloo backend is available. |
| `torch.distributed.is_initialized` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Check if the default process group has been initialized. |
| `torch.distributed.is_mpi_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Check if the MPI backend is available. |
| `torch.distributed.is_nccl_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Check if the NCCL backend is available. |
| `torch.distributed.is_torchelastic_launched` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Check whether this process was launched with ``torch.distributed.elastic`` (aka torchelastic). The existence of ``TORCHELASTIC_RUN_ID`` environment variable is used as a proxy to determine whether ... |
| `torch.distributed.is_ucc_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Check if the UCC backend is available. |
| `torch.distributed.is_xccl_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Check if the XCCL backend is available. |
| `torch.distributed.isend` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, dst, group, ...` | `typing.Optional[torch.distributed.distributed_c10d.Work]` | Send a tensor asynchronously. .. warning:: Modifying ``tensor`` before the request completes causes undefined behavior. .. warning:: ``tag`` is not supported with the NCCL backend. Unlike send, whi... |
| `torch.distributed.monitored_barrier` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group, timeout, wait_all_ranks` | `Any` | Synchronize processes similar to ``torch.distributed.barrier``, but consider a configurable timeout. It is able to report ranks that did not pass this barrier within the provided timeout. Specifica... |
| `torch.distributed.new_group` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ranks, timeout, backend, ...` | `Any` | Create a new distributed group. This function requires that all processes in the main group (i.e. all processes that are part of the distributed job) enter this function, even if they are not going... |
| `torch.distributed.new_subgroups` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group_size, group, timeout, ...` | `Any` | Create subgroups of equal size. By default, it creates intra-machine subgroups, where each of which contains all the ranks of a machine, based on the assumption that each machine has the same numbe... |
| `torch.distributed.new_subgroups_by_enumeration` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ranks_per_subgroup_list, timeout, backend, ...` | `Any` | Create subgroups by dividing the global world. The division is specified by a nested list of ranks. The subgroups cannot have overlap, and some ranks may not have to be in any subgroup. This is a c... |
| `torch.distributed.recv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, src, group, ...` | `<class 'int'>` | Receives a tensor synchronously. .. warning:: ``tag`` is not supported with the NCCL backend. Args: tensor (Tensor): Tensor to fill with received data. src (int, optional): Source rank on global pr... |
| `torch.distributed.recv_object_list` | ❓ | ❓ | ❓ | ❓ | 🔴 | `object_list, src, group, ...` | `Any` | Receives picklable objects in ``object_list`` synchronously. Similar to :func:`recv`, but can receive Python objects. Args: object_list (List[Any]): List of objects to receive into. Must provide a ... |
| `torch.distributed.reduce` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, dst, op, ...` | `Any` | Reduces the tensor data across all machines. Only the process with rank ``dst`` is going to receive the final result. Args: tensor (Tensor): Input and output of the collective. The function operate... |
| `torch.distributed.reduce_scatter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `output, input_list, op, ...` | `Any` | Reduces, then scatters a list of tensors to all processes in a group. Args: output (Tensor): Output tensor. input_list (list[Tensor]): List of tensors to reduce and scatter. op (optional): One of t... |
| `torch.distributed.reduce_scatter_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `output, input, op, ...` | `Any` | Reduces, then scatters a tensor to all ranks in a group. Args: output (Tensor): Output tensor. It should have the same size across all ranks. input (Tensor): Input tensor to be reduced and scattere... |
| `torch.distributed.register_rendezvous_handler` | ❓ | ❓ | ❓ | ❓ | 🔴 | `scheme, handler` | `Any` | Register a new rendezvous handler. Before we can run collective algorithms, participating processes need to find each other and exchange information to be able to communicate. We call this process ... |
| `torch.distributed.rendezvous` | ❓ | ❓ | ❓ | ❓ | 🔴 | `url, rank, world_size, ...` | `Any` |  |
| `torch.distributed.scatter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, scatter_list, src, ...` | `Any` | Scatters a list of tensors to all processes in a group. Each process will receive exactly one tensor and store its data in the ``tensor`` argument. Complex tensors are supported. Args: tensor (Tens... |
| `torch.distributed.scatter_object_list` | ❓ | ❓ | ❓ | ❓ | 🔴 | `scatter_object_output_list, scatter_object_input_list, src, ...` | `Any` | Scatters picklable objects in ``scatter_object_input_list`` to the whole group. Similar to :func:`scatter`, but Python objects can be passed in. On each rank, the scattered object will be stored as... |
| `torch.distributed.send` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, dst, group, ...` | `None` | Send a tensor synchronously. .. warning:: ``tag`` is not supported with the NCCL backend. Args: tensor (Tensor): Tensor to send. dst (int): Destination rank on global process group (regardless of `... |
| `torch.distributed.send_object_list` | ❓ | ❓ | ❓ | ❓ | 🔴 | `object_list, dst, group, ...` | `Any` | Sends picklable objects in ``object_list`` synchronously. Similar to :func:`send`, but Python objects can be passed in. Note that all objects in ``object_list`` must be picklable in order to be sen... |
| `torch.distributed.split_group` | ❓ | ❓ | ❓ | ❓ | 🔴 | `parent_pg, split_ranks, timeout, ...` | `typing.Optional[torch.distributed.distributed_c10d.ProcessGroup]` | Create a new process group splitted from the given parent process group. warning:: This is an experimental API and only the ``NCCL`` backend supports this API. Other backends will raise an error. U... |
| `torch.distributed.supports_complex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `reduceOp` | `<class 'bool'>` | Return true if reduce ops is supported. False otherwise. |
| | | | | | | | | |
| 🟦 DISTRIBUTIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.distributions.kl_divergence` | ❓ | ❓ | ❓ | ❓ | 🔴 | `p, q` | `<class 'torch.Tensor'>` | Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions. .. math:: KL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)} \,dx Args: p (Distribution): A :class:`~torch.distribution... |
| `torch.distributions.register_kl` | ❓ | ❓ | ❓ | ❓ | 🔴 | `type_p, type_q` | `Any` | Decorator to register a pairwise function with :meth:`kl_divergence`. Usage:: @register_kl(Normal, Normal) def kl_normal_normal(p, q): # insert implementation here Lookup returns the most specific ... |
| | | | | | | | | |
| 🟦 MODEL_EXPORT | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.export.compatibility` | ❓ | ❓ | ❓ | ❓ | 🔴 | `is_backward_compatible` | `typing.Callable[[~_T], ~_T]` |  |
| `torch.export.default_decompositions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `CustomDecompTable` | This is the default decomposition table which contains decomposition of all ATEN operators to core aten opset. Use this API together with :func:`run_decompositions()` |
| `torch.export.dims` | ❓ | ❓ | ❓ | ❓ | 🔴 | `names, min, max` | `tuple[torch.export.dynamic_shapes.Dim, ...]` | Util to create multiple :func:`Dim` types. Returns: A tuple of :func:`Dim` types. |
| `torch.export.export` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod, args, kwargs, ...` | `<class 'torch.export.exported_program.ExportedProgram'>` | :func:`export` takes any nn.Module along with example inputs, and produces a traced graph representing only the Tensor computation of the function in an Ahead-of-Time (AOT) fashion, which can subse... |
| `torch.export.export_for_training` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod, args, kwargs, ...` | `<class 'torch.export.exported_program.ExportedProgram'>` | :func:`export_for_training` takes any nn.Module along with example inputs, and produces a traced graph representing only the Tensor computation of the function in an Ahead-of-Time (AOT) fashion, wh... |
| `torch.export.load` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f, extra_files, expected_opset_version` | `<class 'torch.export.exported_program.ExportedProgram'>` | .. warning:: Under active development, saved files may not be usable in newer versions of PyTorch. Loads an :class:`ExportedProgram` previously saved with :func:`torch.export.save <torch.export.sav... |
| `torch.export.register_dataclass` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cls, serialized_type_name` | `None` | Registers a dataclass as a valid input/output type for :func:`torch.export.export`. Args: cls: the dataclass type to register serialized_type_name: The serialized name for the dataclass. This is re... |
| `torch.export.save` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ep, f, extra_files, ...` | `None` | .. warning:: Under active development, saved files may not be usable in newer versions of PyTorch. Saves an :class:`ExportedProgram` to a file-like object. It can then be loaded using the Python AP... |
| `torch.export.unflatten` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module, flat_args_adapter` | `<class 'torch.export.unflatten.UnflattenedModule'>` | Unflatten an ExportedProgram, producing a module with the same module hierarchy as the original eager module. This can be useful if you are trying to use :mod:`torch.export` with another system tha... |
| | | | | | | | | |
| 🟦 FUNCTIONAL_PROGRAMMING | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.func.debug_unwrap` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, recurse` | `<class 'torch.Tensor'>` | Unwraps a functorch tensor (e.g. BatchedTensor, GradTrackingTensor) to its underlying tensor. This function should only be used in a debug setting (e.g. trying to print the value of a Tensor in a d... |
| `torch.func.functional_call` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module, parameter_and_buffer_dicts, args, ...` | `Any` | Performs a functional call on the module by replacing the module parameters and buffers with the provided ones. .. note:: If the module has active parametrizations, passing a value in the :attr:`pa... |
| `torch.func.functionalize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, remove` | `typing.Callable` | functionalize is a transform that can be used to remove (intermediate) mutations and aliasing from a function, while preserving the function's semantics. ``functionalize(func)`` returns a new funct... |
| `torch.func.grad` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, argnums, has_aux` | `typing.Callable` | ``grad`` operator helps computing gradients of ``func`` with respect to the input(s) specified by ``argnums``. This operator can be nested to compute higher-order gradients. Args: func (Callable): ... |
| `torch.func.grad_and_value` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, argnums, has_aux` | `typing.Callable` | Returns a function to compute a tuple of the gradient and primal, or forward, computation. Args: func (Callable): A Python function that takes one or more arguments. Must return a single-element Te... |
| `torch.func.hessian` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, argnums` | `Any` | Computes the Hessian of ``func`` with respect to the arg(s) at index ``argnum`` via a forward-over-reverse strategy. The forward-over-reverse strategy (composing ``jacfwd(jacrev(func))``) is a good... |
| `torch.func.jacfwd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, argnums, has_aux, ...` | `Any` | Computes the Jacobian of ``func`` with respect to the arg(s) at index ``argnum`` using forward-mode autodiff Args: func (function): A Python function that takes one or more arguments, one of which ... |
| `torch.func.jacrev` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, argnums, has_aux, ...` | `Any` | Computes the Jacobian of ``func`` with respect to the arg(s) at index ``argnum`` using reverse mode autodiff .. note:: Using :attr:`chunk_size=1` is equivalent to computing the jacobian row-by-row ... |
| `torch.func.jvp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, primals, tangents, ...` | `Any` | Standing for the Jacobian-vector product, returns a tuple containing the output of `func(*primals)` and the "Jacobian of ``func`` evaluated at ``primals``" times ``tangents``. This is also known as... |
| `torch.func.linearize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, primals` | `tuple[typing.Any, typing.Callable]` | Returns the value of ``func`` at ``primals`` and linear approximation at ``primals``. Args: func (Callable): A Python function that takes one or more arguments. primals (Tensors): Positional argume... |
| `torch.func.replace_all_batch_norm_modules_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `root` | `<class 'torch.nn.modules.module.Module'>` | In place updates :attr:`root` by setting the ``running_mean`` and ``running_var`` to be None and setting track_running_stats to be False for any nn.BatchNorm module in :attr:`root` |
| `torch.func.stack_module_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `models` | `tuple[dict[str, typing.Any], dict[str, typing.Any]]` | stack_module_state(models) -> params, buffers Prepares a list of torch.nn.Modules for ensembling with :func:`vmap`. Given a list of ``M`` ``nn.Modules`` of the same class, returns two dictionaries ... |
| `torch.func.vjp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, primals, has_aux` | `Any` | Standing for the vector-Jacobian product, returns a tuple containing the results of ``func`` applied to ``primals`` and a function that, when given ``cotangents``, computes the reverse-mode Jacobia... |
| `torch.func.vmap` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, in_dims, out_dims, ...` | `typing.Callable` | vmap is the vectorizing map; ``vmap(func)`` returns a new function that maps ``func`` over some dimension of the inputs. Semantically, vmap pushes the map into PyTorch operations called by ``func``... |
| | | | | | | | | |
| 🟦 ASYNCHRONOUS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.futures.cast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typ, val` | `Any` | Cast a value to a type. This returns the value unchanged. To the type checker this signals that the return value has the designated type, but at runtime we intentionally don't check anything (we wa... |
| `torch.futures.collect_all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `futures` | `Future[list[Future]]` | Collects the provided :class:`~torch.futures.Future` objects into a single combined :class:`~torch.futures.Future` that is completed when all of the sub-futures are completed. Args: futures (list):... |
| `torch.futures.wait_all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `futures` | `list` | Waits for all provided futures to be complete, and returns the list of completed values. If any of the futures encounters an error, the method will exit early and report the error not waiting for o... |
| | | | | | | | | |
| 🟦 GRAPH_TRANSFORMATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.fx.has_side_effect` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `typing.Callable[~_P, ~_R]` | .. warning:: This API is experimental and is *NOT* backward-compatible. |
| `torch.fx.map_arg` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, fn` | `~ArgumentT` | Apply fn recursively to each Node appearing in arg. arg may be a list, tuple, slice, or dict with string keys: the return value will have the same type and structure. .. note:: Backwards-compatibil... |
| `torch.fx.replace_pattern` | ❓ | ❓ | ❓ | ❓ | 🔴 | `gm, pattern, replacement` | `list[torch.fx.subgraph_rewriter.Match]` | Matches all possible non-overlapping sets of operators and their data dependencies (``pattern``) in the Graph of a GraphModule (``gm``), then replaces each of these matched subgraphs with another s... |
| `torch.fx.symbolic_trace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `root, concrete_args` | `<class 'torch.fx.graph_module.GraphModule'>` | Symbolic tracing API Given an ``nn.Module`` or function instance ``root``, this function will return a ``GraphModule`` constructed by recording operations seen while tracing through ``root``. ``con... |
| `torch.fx.wrap` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn_or_name` | `Any` | This function can be called at module-level scope to register fn_or_name as a "leaf function". A "leaf function" will be preserved as a CallFunction node in the FX trace instead of being traced thr... |
| | | | | | | | | |
| 🟦 JIT_COMPILATION | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.jit.annotate` | ❓ | ❓ | ❓ | ❓ | 🔴 | `the_type, the_value` | `Any` | Use to give type of `the_value` in TorchScript compiler. This method is a pass-through function that returns `the_value`, used to hint TorchScript compiler the type of `the_value`. It is a no-op wh... |
| `torch.jit.contextmanager` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` | @contextmanager decorator. Typical usage: @contextmanager def some_generator(<arguments>): <setup> try: yield <value> finally: <cleanup> This makes this: with some_generator(<arguments>) as <variab... |
| `torch.jit.enable_onednn_fusion` | ❓ | ❓ | ❓ | ❓ | 🔴 | `enabled` | `Any` | Enable or disables onednn JIT fusion based on the parameter `enabled`. |
| `torch.jit.export` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` | This decorator indicates that a method on an ``nn.Module`` is used as an entry point into a :class:`ScriptModule` and should be compiled. ``forward`` implicitly is assumed to be an entry point, so ... |
| `torch.jit.export_opnames` | ❓ | ❓ | ❓ | ❓ | 🔴 | `m` | `Any` | Generate new bytecode for a Script module. Returns what the op list would be for a Script Module based off the current code base. If you have a LiteScriptModule and want to get the currently presen... |
| `torch.jit.fork` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, args, kwargs` | `Any` | Create an asynchronous task executing `func` and a reference to the value of the result of this execution. `fork` will return immediately, so the return value of `func` may not have been computed y... |
| `torch.jit.freeze` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod, preserved_attrs, optimize_numerics` | `Any` | Freeze ScriptModule, inline submodules, and attributes as constants. Freezing a :class:`ScriptModule` will clone it and attempt to inline the cloned module's submodules, parameters, and attributes ... |
| `torch.jit.fuser` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name` | `Any` | Context manager that facilitates switching between backend fusers. Valid names: * ``fuser0`` - enables only legacy fuser * ``fuser1`` - enables only NNC * ``fuser2`` - enables only nvFuser * ``fuse... |
| `torch.jit.ignore` | ❓ | ❓ | ❓ | ❓ | 🔴 | `drop, kwargs` | `Any` | This decorator indicates to the compiler that a function or method should be ignored and left as a Python function. This allows you to leave code in your model that is not yet TorchScript compatibl... |
| `torch.jit.interface` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj` | `Any` | Decorate to annotate classes or modules of different types. This decorator can be used to define an interface that can be used to annotate classes or modules of different types. This can be used fo... |
| `torch.jit.is_scripting` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Function that returns True when in compilation and False otherwise. This is useful especially with the @unused decorator to leave code in your model that is not yet TorchScript compatible. .. testc... |
| `torch.jit.is_tracing` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return a boolean value. Returns ``True`` in tracing (if a function is called during the tracing of code with ``torch.jit.trace``) and ``False`` otherwise. |
| `torch.jit.isinstance` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj, target_type` | `Any` | Provide container type refinement in TorchScript. It can refine parameterized containers of the List, Dict, Tuple, and Optional types. E.g. ``List[str]``, ``Dict[str, List[torch.Tensor]]``, ``Optio... |
| `torch.jit.jit_module_from_flatbuffer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f` | `Any` |  |
| `torch.jit.load` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f, map_location, _extra_files, ...` | `Any` | Load a :class:`ScriptModule` or :class:`ScriptFunction` previously saved with :func:`torch.jit.save <torch.jit.save>`. All previously saved modules, no matter their device, are first loaded onto CP... |
| `torch.jit.onednn_fusion_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return whether onednn JIT fusion is enabled. |
| `torch.jit.optimize_for_inference` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod, other_methods` | `<class 'torch.jit._script.ScriptModule'>` | Perform a set of optimization passes to optimize a model for the purposes of inference. If the model is not already frozen, optimize_for_inference will invoke `torch.jit.freeze` automatically. In a... |
| `torch.jit.optimized_execution` | ❓ | ❓ | ❓ | ❓ | 🔴 | `should_optimize` | `Any` | Context manager that controls whether the JIT's executor will run optimizations before executing a function. |
| `torch.jit.run_frozen_optimizations` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod, optimize_numerics, preserved_methods` | `Any` | Run a series of optimizations looking for patterns that occur in frozen graphs. The current set of optimizations includes: - Dropout Removal - Pretranspose Linear Layers - Concat Linear Layers with... |
| `torch.jit.save` | ❓ | ❓ | ❓ | ❓ | 🔴 | `m, f, _extra_files` | `Any` | Save an offline version of this module for use in a separate process. The saved module serializes all of the methods, submodules, parameters, and attributes of this module. It can be loaded into th... |
| `torch.jit.save_jit_module_to_flatbuffer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `m, f, _extra_files` | `Any` | Save an offline version of this module for use in a separate process. The saved module serializes all of the methods, submodules, parameters, and attributes of this module. It can be loaded into th... |
| `torch.jit.script` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj, optimize, _frames_up, ...` | `Any` | Script the function. Scripting a function or ``nn.Module`` will inspect the source code, compile it as TorchScript code using the TorchScript compiler, and return a :class:`ScriptModule` or :class:... |
| `torch.jit.script_if_tracing` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` | Compiles ``fn`` when it is first called during tracing. ``torch.jit.script`` has a non-negligible start up time when it is first called due to lazy-initializations of many compiler builtins. Theref... |
| `torch.jit.script_method` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` |  |
| `torch.jit.set_fusion_strategy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `strategy` | `Any` | Set the type and number of specializations that can occur during fusion. Usage: provide a list of pairs (type, depth) where type is one of "STATIC" or "DYNAMIC" and depth is an integer. Behavior - ... |
| `torch.jit.set_module` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj, mod` | `Any` | Set the module attribute on a python object for a given object for nicer printing |
| `torch.jit.trace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, example_inputs, optimize, ...` | `Any` | Trace a function and return an executable or :class:`ScriptFunction` that will be optimized using just-in-time compilation. Tracing is ideal for code that operates only on ``Tensor``\\s and lists, ... |
| `torch.jit.trace_module` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod, inputs, optimize, ...` | `Any` | Trace a module and return an executable :class:`ScriptModule` that will be optimized using just-in-time compilation. When a module is passed to :func:`torch.jit.trace <torch.jit.trace>`, only the `... |
| `torch.jit.unused` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` | This decorator indicates to the compiler that a function or method should be ignored and replaced with the raising of an exception. This allows you to leave code in your model that is not yet Torch... |
| `torch.jit.wait` | ❓ | ❓ | ❓ | ❓ | 🔴 | `future` | `Any` | Force completion of a `torch.jit.Future[T]` asynchronous task, returning the result of the task. See :func:`~fork` for docs and examples. Args: future (torch.jit.Future[T]): an asynchronous task re... |
| | | | | | | | | |
| 🟦 MASKED_OPERATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.masked.amax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, keepdim, ...` | `<class 'torch.Tensor'>` | amax(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor Returns maximum of all the elements in the :attr:`input` tensor along the given dimension(s) :attr:`dim` while the :attr:`input` ... |
| `torch.masked.amin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, keepdim, ...` | `<class 'torch.Tensor'>` | amin(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor Returns minimum of all the elements in the :attr:`input` tensor along the given dimension(s) :attr:`dim` while the :attr:`input` ... |
| `torch.masked.argmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, keepdim, ...` | `<class 'torch.Tensor'>` | argmax(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor Returns argmax of all the elements in the :attr:`input` tensor along the given dimension(s) :attr:`dim` while the :attr:`input`... |
| `torch.masked.argmin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, keepdim, ...` | `<class 'torch.Tensor'>` | argmin(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor Returns argmin of all the elements in the :attr:`input` tensor along the given dimension(s) :attr:`dim` while the :attr:`input`... |
| `torch.masked.as_masked_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `data, mask` | `<class 'torch.masked.maskedtensor.core.MaskedTensor'>` |  |
| `torch.masked.cumprod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, dtype, ...` | `<class 'torch.Tensor'>` | cumprod(input, dim, *, dtype=None, mask=None) -> Tensor Returns cumulative_prod of all the slices in the :attr:`input` tensor along :attr:`dim` while the :attr:`input` elements are masked out accor... |
| `torch.masked.cumsum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, dtype, ...` | `<class 'torch.Tensor'>` | cumsum(input, dim, *, dtype=None, mask=None) -> Tensor Returns cumulative_sum of all the slices in the :attr:`input` tensor along :attr:`dim` while the :attr:`input` elements are masked out accordi... |
| `torch.masked.is_masked_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj` | `typing_extensions.TypeIs[ForwardRef('MaskedTensor')]` | Returns True if the input is a MaskedTensor, else False Args: a: any input Examples: >>> # xdoctest: +SKIP >>> from torch.masked import MaskedTensor >>> data = torch.arange(6).reshape(2,3) >>> mask... |
| `torch.masked.log_softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, dtype, ...` | `<class 'torch.Tensor'>` | log_softmax(input, dim, *, dtype=None, mask=None) -> Tensor Returns log_softmax of all the slices in the :attr:`input` tensor along :attr:`dim` while the :attr:`input` elements are masked out accor... |
| `torch.masked.logaddexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, other, dtype, ...` | `<class 'torch.Tensor'>` | logaddexp(input, other, *, dtype=None, input_mask=None, other_mask=None) -> Tensor Returns logaddexp of all the elements in the :attr:`input` and the :attr:`other` tensor. The :attr:`input` element... |
| `torch.masked.logsumexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, keepdim, ...` | `<class 'torch.Tensor'>` | logsumexp(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor Returns logsumexp of all the elements in the :attr:`input` tensor along the given dimension(s) :attr:`dim` while the :attr:`... |
| `torch.masked.masked_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `data, mask, requires_grad` | `<class 'torch.masked.maskedtensor.core.MaskedTensor'>` |  |
| `torch.masked.mean` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, keepdim, ...` | `<class 'torch.Tensor'>` | mean(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor Returns mean of all the elements in the :attr:`input` tensor along the given dimension(s) :attr:`dim` while the :attr:`input` ele... |
| `torch.masked.median` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, keepdim, ...` | `<class 'torch.Tensor'>` | median(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor Returns median of all the elements in the :attr:`input` tensor along the given dimension(s) :attr:`dim` while the :attr:`input`... |
| `torch.masked.norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, ord, dim, ...` | `<class 'torch.Tensor'>` | norm(input, ord, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor Returns norm of all the elements in the :attr:`input` tensor along the given dimension(s) :attr:`dim` while the :attr:`input... |
| `torch.masked.normalize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, ord, dim, ...` | `<class 'torch.Tensor'>` | normalize(input, ord, dim, *, eps=1e-12, dtype=None, mask=None) -> Tensor Returns normalize of all the slices in the :attr:`input` tensor along :attr:`dim` while the :attr:`input` elements are mask... |
| `torch.masked.prod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, keepdim, ...` | `<class 'torch.Tensor'>` | prod(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor Returns product of all the elements in the :attr:`input` tensor along the given dimension(s) :attr:`dim` while the :attr:`input` ... |
| `torch.masked.softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, dtype, ...` | `<class 'torch.Tensor'>` | softmax(input, dim, *, dtype=None, mask=None) -> Tensor Returns softmax of all the slices in the :attr:`input` tensor along :attr:`dim` while the :attr:`input` elements are masked out according to ... |
| `torch.masked.softmin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, dtype, ...` | `<class 'torch.Tensor'>` | softmin(input, dim, *, dtype=None, mask=None) -> Tensor Returns softmin of all the slices in the :attr:`input` tensor along :attr:`dim` while the :attr:`input` elements are masked out according to ... |
| `torch.masked.std` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, unbiased, ...` | `<class 'torch.Tensor'>` | std(input, dim, unbiased, *, keepdim=False, dtype=None, mask=None) -> Tensor Returns standard_deviation of all the elements in the :attr:`input` tensor along the given dimension(s) :attr:`dim` whil... |
| `torch.masked.sum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, keepdim, ...` | `<class 'torch.Tensor'>` | sum(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor Returns sum of all the elements in the :attr:`input` tensor along the given dimension(s) :attr:`dim` while the :attr:`input` eleme... |
| `torch.masked.var` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, unbiased, ...` | `<class 'torch.Tensor'>` | var(input, dim, unbiased, *, keepdim=False, dtype=None, mask=None) -> Tensor Returns variance of all the elements in the :attr:`input` tensor along the given dimension(s) :attr:`dim` while the :att... |
| | | | | | | | | |
| 🟦 MPS_OPERATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.mps.compile_shader` | ❓ | ❓ | ❓ | ❓ | 🔴 | `source` | `Any` | Compiles compute shader from source and allows one to invoke kernels defined there from the comfort of Python runtime Example:: >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_MPS) >>> lib = torch.mps.... |
| `torch.mps.current_allocated_memory` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Returns the current GPU memory occupied by tensors in bytes. .. note:: The returned size does not include cached allocations in memory pools of MPSAllocator. |
| `torch.mps.device_count` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Returns the number of available MPS devices. |
| `torch.mps.driver_allocated_memory` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Returns total GPU memory allocated by Metal driver for the process in bytes. .. note:: The returned size includes cached allocations in MPSAllocator pools as well as allocations from MPS/MPSGraph f... |
| `torch.mps.empty_cache` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU applications. |
| `torch.mps.get_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.Tensor'>` | Returns the random number generator state as a ByteTensor. Args: device (torch.device or int, optional): The device to return the RNG state of. Default: ``'mps'`` (i.e., ``torch.device('mps')``, th... |
| `torch.mps.is_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` |  |
| `torch.mps.manual_seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `seed` | `None` | Sets the seed for generating random numbers. Args: seed (int): The desired seed. |
| `torch.mps.recommended_max_memory` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Returns recommended max Working set size for GPU memory in bytes. .. note:: Recommended max working set size for Metal. returned from device.recommendedMaxWorkingSetSize. |
| `torch.mps.seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Sets the seed for generating random numbers to a random number. |
| `torch.mps.set_per_process_memory_fraction` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fraction` | `None` | Set memory fraction for limiting process's memory allocation on MPS device. The allowed value equals the fraction multiplied by recommended maximum device memory (obtained from Metal API device.rec... |
| `torch.mps.set_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `new_state, device` | `None` | Sets the random number generator state. Args: new_state (torch.ByteTensor): The desired state device (torch.device or int, optional): The device to set the RNG state. Default: ``'mps'`` (i.e., ``to... |
| `torch.mps.synchronize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Waits for all kernels in all streams on a MPS device to complete. |
| | | | | | | | | |
| 🟦 MTIA_OPERATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.mtia.classproperty` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` |  |
| `torch.mtia.current_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the index of a currently selected device. |
| `torch.mtia.current_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.Stream'>` | Return the currently selected :class:`Stream` for a given device. Args: device (torch.device or int, optional): selected device. Returns the currently selected :class:`Stream` for the current devic... |
| `torch.mtia.default_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.Stream'>` | Return the default :class:`Stream` for a given device. Args: device (torch.device or int, optional): selected device. Returns the default :class:`Stream` for the current device, given by :func:`~to... |
| `torch.mtia.device_count` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the number of MTIA devices available. |
| `torch.mtia.empty_cache` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Empty the MTIA device cache. |
| `torch.mtia.get_device_capability` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `tuple[int, int]` | Return capability of a given device as a tuple of (major version, minor version). Args: device (torch.device or int, optional) selected device. Returns statistics for the current device, given by c... |
| `torch.mtia.get_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.Tensor'>` | Returns the random number generator state as a ByteTensor. Args: device (torch.device or int, optional): The device to return the RNG state of. Default: ``'mtia'`` (i.e., ``torch.device('mtia')``, ... |
| `torch.mtia.init` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.mtia.is_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Return true if MTIA device is available |
| `torch.mtia.is_initialized` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return whether PyTorch's MTIA state has been initialized. |
| `torch.mtia.max_memory_allocated` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the maximum memory allocated in bytes for a given device. Args: device (torch.device, str, or int, optional) selected device. Returns statistics for the current device, given by current_devi... |
| `torch.mtia.memory_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `dict[str, typing.Any]` | Return a dictionary of MTIA memory allocator statistics for a given device. Args: device (torch.device, str, or int, optional) selected device. Returns statistics for the current device, given by c... |
| `torch.mtia.record_memory_history` | ❓ | ❓ | ❓ | ❓ | 🔴 | `enabled, stacks, max_entries` | `None` | Enable/Disable the memory profiler on MTIA allocator Args: enabled (all or state, optional) selected device. Returns statistics for the current device, given by current_device(), if device is None ... |
| `torch.mtia.reset_peak_memory_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Reset the peak memory stats for a given device. Args: device (torch.device, str, or int, optional) selected device. Returns statistics for the current device, given by current_device(), if device i... |
| `torch.mtia.set_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Set the current device. Args: device (torch.device or int): selected device. This function is a no-op if this argument is negative. |
| `torch.mtia.set_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `new_state, device` | `None` | Sets the random number generator state. Args: new_state (torch.ByteTensor): The desired state device (torch.device or int, optional): The device to set the RNG state. Default: ``'mtia'`` (i.e., ``t... |
| `torch.mtia.set_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `Any` | Set the current stream.This is a wrapper API to set the stream. Usage of this function is discouraged in favor of the ``stream`` context manager. Args: stream (Stream): selected stream. This functi... |
| `torch.mtia.snapshot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `dict[str, typing.Any]` | Return a dictionary of MTIA memory allocator history |
| `torch.mtia.stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `<class 'torch.mtia.StreamContext'>` | Wrap around the Context-manager StreamContext that selects a given stream. Arguments: stream (Stream): selected stream. This manager is a no-op if it's ``None``. .. note:: In eager mode stream is o... |
| `torch.mtia.synchronize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Waits for all jobs in all streams on a MTIA device to complete. |
| | | | | | | | | |
| 🟦 MULTIPROCESSING | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.multiprocessing.active_children` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return list of process objects corresponding to live child processes |
| `torch.multiprocessing.current_process` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return process object representing the current process |
| `torch.multiprocessing.get_all_sharing_strategies` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return a set of sharing strategies supported on a current system. |
| `torch.multiprocessing.get_sharing_strategy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return the current strategy for sharing CPU tensors. |
| `torch.multiprocessing.init_reductions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.multiprocessing.parent_process` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return process object representing the parent process |
| `torch.multiprocessing.set_sharing_strategy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `new_strategy` | `Any` | Set the strategy for sharing CPU tensors. Args: new_strategy (str): Name of the selected strategy. Should be one of the values returned by :func:`get_all_sharing_strategies()`. |
| `torch.multiprocessing.spawn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn, args, nprocs, ...` | `Any` | Spawns ``nprocs`` processes that run ``fn`` with ``args``. If one of the processes exits with a non-zero exit status, the remaining processes are killed and an exception is raised with the cause of... |
| `torch.multiprocessing.start_processes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn, args, nprocs, ...` | `Any` |  |
| | | | | | | | | |
| 🟦 NESTED_TENSORS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.nested.as_nested_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ts, dtype, device, ...` | `<class 'torch.Tensor'>` | Constructs a nested tensor preserving autograd history from a tensor or a list / tuple of tensors. If a nested tensor is passed, it will be returned directly unless the device / dtype / layout diff... |
| `torch.nested.masked_select` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, mask` | `<class 'torch.Tensor'>` | Constructs a nested tensor given a strided tensor input and a strided mask, the resulting jagged layout nested tensor will have values retain values where the mask is equal to True. The dimensional... |
| `torch.nested.narrow` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, dim, start, ...` | `<class 'torch.Tensor'>` | Constructs a nested tensor (which might be a view) from :attr:`tensor`, a strided tensor. This follows similar semantics to torch.Tensor.narrow, where in the :attr:`dim`-th dimension the new nested... |
| `torch.nested.nested_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor_list, dtype, layout, ...` | `<class 'torch.Tensor'>` | Constructs a nested tensor with no autograd history (also known as a "leaf tensor", see :ref:`Autograd mechanics <autograd-mechanics>`) from :attr:`tensor_list` a list of tensors. Args: tensor_list... |
| `torch.nested.nested_tensor_from_jagged` | ❓ | ❓ | ❓ | ❓ | 🔴 | `values, offsets, lengths, ...` | `<class 'torch.Tensor'>` | Constructs a jagged layout nested tensor from the given jagged components. The jagged layout consists of a required values buffer with the jagged dimension packed into a single dimension. The offse... |
| | | | | | | | | |
| 🟦 NEURAL_NETWORK | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.nn.factory_kwargs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kwargs` | `Any` | Return a canonicalized dict of factory kwargs. Given kwargs, returns a canonicalized dict of factory kwargs that can be directly passed to factory functions like torch.empty, or errors if unrecogni... |
| | | | | | | | | |
| 🟦 ONNX_EXPORT | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.onnx.dynamo_export` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, model_args, export_options, ...` | `ONNXProgram` | Export a torch.nn.Module to an ONNX graph. .. deprecated:: 2.7 Please use ``torch.onnx.export(..., dynamo=True)`` instead. Args: model: The PyTorch model to be exported to ONNX. model_args: Positio... |
| `torch.onnx.enable_fake_mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Enable fake mode for the duration of the context. Internally it instantiates a :class:`torch._subclasses.fake_tensor.FakeTensorMode` context manager that converts user input and model parameters in... |
| `torch.onnx.export` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, args, f, ...` | `ONNXProgram | None` | Exports a model into ONNX format. Setting ``dynamo=True`` enables the new ONNX export logic which is based on :class:`torch.export.ExportedProgram` and a more modern set of translation logic. This ... |
| `torch.onnx.is_in_onnx_export` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `bool` | Returns whether it is in the middle of ONNX export. |
| `torch.onnx.is_onnxrt_backend_supported` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Returns ``True`` if ONNX Runtime dependencies are installed and usable to support TorchDynamo backend integration; ``False`` otherwise. Example:: # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX) >>> i... |
| `torch.onnx.register_custom_op_symbolic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `symbolic_name, symbolic_fn, opset_version` | `Any` | Registers a symbolic function for a custom operator. When the user registers symbolic for custom/contrib ops, it is highly recommended to add shape inference for that operator via setType API, othe... |
| `torch.onnx.select_model_mode_for_export` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, mode` | `Any` | A context manager to temporarily set the training mode of ``model`` to ``mode``, resetting it when we exit the with-block. .. deprecated:: 2.7 Please set training mode before exporting the model. A... |
| `torch.onnx.unregister_custom_op_symbolic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `symbolic_name, opset_version` | `Any` | Unregisters ``symbolic_name``. See "Custom Operators" in the module documentation for an example usage. Args: symbolic_name (str): The name of the custom operator in "<domain>::<op>" format. opset_... |
| | | | | | | | | |
| 🟦 PACKAGING | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.package.is_from_package` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj` | `<class 'bool'>` | Return whether an object was loaded from a package. Note: packaged objects from externed modules will return ``False``. |
| | | | | | | | | |
| 🟦 PROFILING | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.profiler.is_fbcode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` |  |
| `torch.profiler.register_optimizer_step_post_hook` | ❓ | ❓ | ❓ | ❓ | 🔴 | `hook` | `<class 'torch.utils.hooks.RemovableHandle'>` | Register a post hook common to all optimizers. The hook should have the following signature:: hook(optimizer, args, kwargs) -> None Args: hook (Callable): A user defined hook which is registered on... |
| `torch.profiler.schedule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `wait, warmup, active, ...` | `typing.Callable` | Returns a callable that can be used as profiler ``schedule`` argument. The profiler will skip the first ``skip_first`` steps, then wait for ``wait`` steps, then do the warmup for the next ``warmup`... |
| `torch.profiler.supported_activities` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Returns a set of supported profiler tracing activities. Note: profiler uses CUPTI library to trace on-device CUDA kernels. In case when CUDA is enabled but CUPTI is not available, passing ``Profile... |
| `torch.profiler.tensorboard_trace_handler` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dir_name, worker_name, use_gzip` | `Any` | Outputs tracing files to directory of ``dir_name``, then that directory can be directly delivered to tensorboard as logdir. ``worker_name`` should be unique for each worker in distributed scenario,... |
| | | | | | | | | |
| 🟦 QUANTIZATION | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.quantization.add_quant_dequant` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module` | `Any` | Wrap the leaf child module in QuantWrapper if it has a valid qconfig Note that this function will modify the children of module inplace and it can return a new module which wraps the input module a... |
| `torch.quantization.convert` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module, mapping, inplace, ...` | `Any` | Converts submodules in input module to a different module according to `mapping` by calling `from_float` method on the target module class. And remove qconfig at the end if remove_qconfig is set to... |
| `torch.quantization.convert_dynamic_jit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, inplace, debug, ...` | `Any` |  |
| `torch.quantization.convert_jit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, inplace, debug, ...` | `Any` |  |
| `torch.quantization.default_eval_fn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, calib_data` | `Any` | Default evaluation function takes a torch.utils.data.Dataset or a list of input Tensors and run the model on the dataset |
| `torch.quantization.disable_fake_quant` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod` | `Any` | Disable fake quantization for the module. Disable fake quantization for this module, if applicable. Example usage:: # model is any PyTorch model model.apply(torch.ao.quantization.disable_fake_quant) |
| `torch.quantization.disable_observer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod` | `Any` | Disable observation for this module. Disable observation for this module, if applicable. Example usage:: # model is any PyTorch model model.apply(torch.ao.quantization.disable_observer) |
| `torch.quantization.enable_fake_quant` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod` | `Any` | Enable fake quantization for the module. Enable fake quantization for this module, if applicable. Example usage:: # model is any PyTorch model model.apply(torch.ao.quantization.enable_fake_quant) |
| `torch.quantization.enable_observer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod` | `Any` | Enable observation for this module. Enable observation for this module, if applicable. Example usage:: # model is any PyTorch model model.apply(torch.ao.quantization.enable_observer) |
| `torch.quantization.fuse_conv_bn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `is_qat, conv, bn` | `Any` | Return the fused the conv and bn modules. Given the conv and bn modules, fuses them and returns the fused module Args: is_qat: a flag for whether we are using quantization aware training fusion or ... |
| `torch.quantization.fuse_conv_bn_jit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, inplace` | `Any` | Fuse conv - bn module Works for eval model only. Args: model: TorchScript model from scripting or tracing |
| `torch.quantization.fuse_conv_bn_relu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `is_qat, conv, bn, ...` | `Any` | Return the fused conv and bv modules. Given the conv and bn modules, fuses them and returns the fused module Args: is_qat: a flag for whether we are using quantization aware training fusion or post... |
| `torch.quantization.fuse_linear_bn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `is_qat, linear, bn` | `Any` | Return the fused linear and bn modules. Given the linear and bn modules, fuses them and returns the fused module Args: is_qat: a flag for whether we are using quantization aware training fusion or ... |
| `torch.quantization.fuse_modules` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, modules_to_fuse, inplace, ...` | `Any` | Fuse a list of modules into a single module. Fuses only the following sequence of modules: conv, bn conv, bn, relu conv, relu linear, relu bn, relu All other sequences are left unchanged. For these... |
| `torch.quantization.get_default_compare_output_module_list` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `set[typing.Callable]` | Get list of module class types that we will record output in numeric suite |
| `torch.quantization.get_default_dynamic_quant_module_mappings` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `dict[typing.Callable, typing.Any]` | Get module mapping for post training dynamic quantization |
| `torch.quantization.get_default_float_to_quantized_operator_mappings` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `dict[typing.Union[typing.Callable, str], typing.Callable]` |  |
| `torch.quantization.get_default_qat_module_mappings` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `dict[typing.Callable, typing.Any]` | Get default module mapping for quantization aware training |
| `torch.quantization.get_default_qat_qconfig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `backend, version` | `Any` | Returns the default QAT qconfig for the specified backend. Args: * `backend` (str): a string representing the target backend. Currently supports `x86` (default), `fbgemm`, `qnnpack` and `onednn`. *... |
| `torch.quantization.get_default_qconfig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `backend, version` | `Any` | Returns the default PTQ qconfig for the specified backend. Args: * `backend` (str): a string representing the target backend. Currently supports `x86` (default), `fbgemm`, `qnnpack` and `onednn`. R... |
| `torch.quantization.get_default_qconfig_propagation_list` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `set[typing.Callable]` | Get the default list of module types that we'll attach qconfig attribute to in prepare |
| `torch.quantization.get_default_static_quant_module_mappings` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `dict[typing.Callable, typing.Any]` | Get module mapping for post training static quantization |
| `torch.quantization.get_dynamic_quant_module_class` | ❓ | ❓ | ❓ | ❓ | 🔴 | `float_module_class, additional_dynamic_quant_mapping` | `typing.Any` | n Get the dynamically quantized module class corresponding to the floating point module class |
| `torch.quantization.get_fuser_method` | ❓ | ❓ | ❓ | ❓ | 🔴 | `op_list, additional_fuser_method_mapping` | `Any` | Get fuser method for the given list of module types. Get fuser method for the given list of module types, return None if fuser method does not exist |
| `torch.quantization.get_observer_state_dict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod` | `Any` | Returns the state dict corresponding to the observer stats. Traverse the model state_dict and extract out the stats. |
| `torch.quantization.get_quantized_operator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `float_op` | `typing.Callable` | Get the quantized operator corresponding to the float operator |
| `torch.quantization.get_static_quant_module_class` | ❓ | ❓ | ❓ | ❓ | 🔴 | `float_module_class, additional_static_quant_mapping, is_reference` | `typing.Any` | n Get the statically quantized module class corresponding to the floating point module class |
| `torch.quantization.load_observer_state_dict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod, obs_dict` | `Any` | Given input model and a state_dict containing model observer stats, load the stats back into the model. The observer state_dict can be saved using torch.ao.quantization.get_observer_state_dict |
| `torch.quantization.no_observer_set` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `set[typing.Any]` | These modules cannot have observers inserted by default. |
| `torch.quantization.prepare` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, inplace, allow_list, ...` | `Any` | Prepares a copy of the model for quantization calibration or quantization-aware training. Quantization configuration should be assigned preemptively to individual submodules in `.qconfig` attribute... |
| `torch.quantization.prepare_dynamic_jit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, qconfig_dict, inplace` | `Any` |  |
| `torch.quantization.prepare_jit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, qconfig_dict, inplace` | `Any` |  |
| `torch.quantization.prepare_qat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, mapping, inplace` | `Any` | Prepares a copy of the model for quantization calibration or quantization-aware training and converts it to quantized version. Quantization configuration should be assigned preemptively to individu... |
| `torch.quantization.propagate_qconfig_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module, qconfig_dict, prepare_custom_config_dict` | `Any` | Propagate qconfig through the module hierarchy and assign `qconfig` attribute on each leaf module Args: module: input module qconfig_dict: dictionary that maps from name or type of submodule to qua... |
| `torch.quantization.qconfig_equals` | ❓ | ❓ | ❓ | ❓ | 🔴 | `q1, q2` | `Any` | Returns `True` if `q1` equals `q2`, and `False` otherwise. |
| `torch.quantization.quantize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, run_fn, run_args, ...` | `Any` | Quantize the input float model with post training static quantization. First it will prepare the model for calibration, then it calls `run_fn` which will run the calibration step, after that we wil... |
| `torch.quantization.quantize_dynamic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, qconfig_spec, dtype, ...` | `Any` | Converts a float model to dynamic (i.e. weights-only) quantized model. Replaces specified modules with dynamic weight-only quantized versions and output the quantized model. For simplest usage prov... |
| `torch.quantization.quantize_dynamic_jit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, qconfig_dict, inplace, ...` | `Any` | Quantize the input float TorchScript model with post training dynamic quantization. Currently only qint8 quantization of torch.nn.Linear is supported. Args: `model`: input float TorchScript model `... |
| `torch.quantization.quantize_jit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, qconfig_dict, run_fn, ...` | `Any` | Quantize the input float TorchScript model with post training static quantization. First it will prepare the model for calibration, then it calls `run_fn` which will run the calibration step, after... |
| `torch.quantization.quantize_qat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, run_fn, run_args, ...` | `Any` | Do quantization aware training and output a quantized model Args: model: input model run_fn: a function for evaluating the prepared model, can be a function that simply runs the prepared model or a... |
| `torch.quantization.script_qconfig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `qconfig` | `Any` | Instantiate the activation and weight observer modules and script them, these observer module instances will be deepcopied during prepare_jit step. |
| `torch.quantization.script_qconfig_dict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `qconfig_dict` | `Any` | Helper function used by `prepare_jit`. Apply `script_qconfig` for all entries in `qconfig_dict` that is not None. |
| `torch.quantization.swap_module` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod, mapping, custom_module_class_mapping, ...` | `Any` | Swaps the module if it has a quantized counterpart and it has an `observer` attached. Args: mod: input module mapping: a dictionary that maps from nn module to nnq module Return: The corresponding ... |
| | | | | | | | | |
| 🟦 SPARSE_OPERATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.sparse.as_sparse_gradcheck` | ❓ | ❓ | ❓ | ❓ | 🔴 | `gradcheck` | `Any` | Decorate function, to extend gradcheck for sparse tensors. Decorator for torch.autograd.gradcheck or its functools.partial variants that extends the gradcheck function with support to input functio... |
| `torch.sparse.sum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, dtype` | `<class 'torch.Tensor'>` | Return the sum of each row of the given sparse tensor. Returns the sum of each row of the sparse tensor :attr:`input` in the given dimensions :attr:`dim`. If :attr:`dim` is a list of dimensions, re... |
| `torch.sparse.to_sparse_semi_structured` | ❓ | ❓ | ❓ | ❓ | 🔴 | `original_tensor, transposed` | `<class 'torch.sparse.semi_structured.SparseSemiStructuredTensor'>` | This function converts a dense tensor into a sparse semi-structured tensor. It will return a SparseSemiStructuredTensor, a subclass of torch.Tensor. This function will check to ensure the dense ten... |
| | | | | | | | | |
| 🟦 TESTING_UTILITIES | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.testing.assert_allclose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `actual, expected, rtol, ...` | `None` | .. warning:: :func:`torch.testing.assert_allclose` is deprecated since ``1.12`` and will be removed in a future release. Please use :func:`torch.testing.assert_close` instead. You can find detailed... |
| `torch.testing.assert_close` | ❓ | ❓ | ❓ | ❓ | 🔴 | `actual, expected, allow_subclasses, ...` | `Any` | Asserts that ``actual`` and ``expected`` are close. If ``actual`` and ``expected`` are strided, non-quantized, real-valued, and finite, they are considered close if .. math:: \lvert \text{actual} -... |
| `torch.testing.make_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, dtype, device, ...` | `<class 'torch.Tensor'>` | Creates a tensor with the given :attr:`shape`, :attr:`device`, and :attr:`dtype`, and filled with values uniformly drawn from ``[low, high)``. If :attr:`low` or :attr:`high` are specified and are o... |
| | | | | | | | | |
| 🟦 UTILITIES | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.utils.generate_methods_for_privateuse1_backend` | ❓ | ❓ | ❓ | ❓ | 🔴 | `for_tensor, for_module, for_packed_sequence, ...` | `None` | Automatically generate attributes and methods for the custom backend after rename privateuse1 backend. In the default scenario, storage-related methods will not be generated automatically. When you... |
| `torch.utils.get_cpp_backtrace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `frames_to_skip, maximum_number_of_frames` | `<class 'str'>` | Return a string containing the C++ stack trace of the current thread. Args: frames_to_skip (int): the number of frames to skip from the top of the stack maximum_number_of_frames (int): the maximum ... |
| `torch.utils.rename_privateuse1_backend` | ❓ | ❓ | ❓ | ❓ | 🔴 | `backend_name` | `None` | Rename the privateuse1 backend device to make it more convenient to use as a device name within PyTorch APIs. The steps are: (1) (In C++) implement kernels for various torch operations, and registe... |
| `torch.utils.set_module` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj, mod` | `Any` | Set the module attribute on a python object for a given object for nicer printing |
| `torch.utils.swap_tensors` | ❓ | ❓ | ❓ | ❓ | 🔴 | `t1, t2` | `Any` | This function swaps the content of the two Tensor objects. At a high level, this will make t1 have the content of t2 while preserving its identity. This will not work if t1 and t2 have different sl... |
| | | | | | | | | |
| 🟦 XPU_OPERATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.xpu.current_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the index of a currently selected device. |
| `torch.xpu.current_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.xpu.streams.Stream'>` | Return the currently selected :class:`Stream` for a given device. Args: device (torch.device or int, optional): selected device. Returns the currently selected :class:`Stream` for the current devic... |
| `torch.xpu.empty_cache` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Release all unoccupied cached memory currently held by the caching allocator so that those can be used in other XPU application. .. note:: :func:`~torch.xpu.empty_cache` doesn't increase the amount... |
| `torch.xpu.get_arch_list` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `list[str]` | Return list XPU architectures this library was compiled for. |
| `torch.xpu.get_device_name` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'str'>` | Get the name of a device. Args: device (torch.device or int or str, optional): device for which to return the name. This function is a no-op if this argument is a negative integer. It uses the curr... |
| `torch.xpu.get_device_properties` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch._utils._XpuDeviceProperties'>` | Get the properties of a device. Args: device (torch.device or int or str): device for which to return the properties of the device. Returns: _XpuDeviceProperties: the properties of the device |
| `torch.xpu.get_gencode_flags` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'str'>` | Return XPU AOT(ahead-of-time) build flags this library was compiled with. |
| `torch.xpu.get_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.Tensor'>` | Return the random number generator state of the specified GPU as a ByteTensor. Args: device (torch.device or int, optional): The device to return the RNG state of. Default: ``'xpu'`` (i.e., ``torch... |
| `torch.xpu.get_rng_state_all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `list[torch.Tensor]` | Return a list of ByteTensor representing the random number states of all devices. |
| `torch.xpu.get_stream_from_external` | ❓ | ❓ | ❓ | ❓ | 🔴 | `data_ptr, device` | `<class 'torch.xpu.streams.Stream'>` | Return a :class:`Stream` from an external SYCL queue. This function is used to wrap SYCL queue created in other libraries in order to facilitate data exchange and multi-library interactions. .. not... |
| `torch.xpu.init` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Initialize PyTorch's XPU state. This is a Python API about lazy initialization that avoids initializing XPU until the first time it is accessed. Does nothing if the XPU state is already initialized. |
| `torch.xpu.initial_seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the current random seed of the current GPU. .. warning:: This function eagerly initializes XPU. |
| `torch.xpu.is_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Return a bool indicating if XPU is currently available. |
| `torch.xpu.is_bf16_supported` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return a bool indicating if the current XPU device supports dtype bfloat16. |
| `torch.xpu.is_initialized` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return whether PyTorch's XPU state has been initialized. |
| `torch.xpu.lru_cache` | ❓ | ❓ | ❓ | ❓ | 🔴 | `maxsize, typed` | `Any` | Least-recently-used cache decorator. If *maxsize* is set to None, the LRU features are disabled and the cache can grow without bound. If *typed* is True, arguments of different types will be cached... |
| `torch.xpu.manual_seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `seed` | `None` | Set the seed for generating random numbers for the current GPU. It's safe to call this function if XPU is not available; in that case, it is silently ignored. Args: seed (int): The desired seed. ..... |
| `torch.xpu.manual_seed_all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `seed` | `None` | Set the seed for generating random numbers on all GPUs. It's safe to call this function if XPU is not available; in that case, it is silently ignored. Args: seed (int): The desired seed. |
| `torch.xpu.max_memory_allocated` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the maximum GPU memory occupied by tensors in bytes for a given device. By default, this returns the peak allocated memory since the beginning of this program. :func:`~torch.xpu.reset_peak_m... |
| `torch.xpu.max_memory_reserved` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the maximum GPU memory managed by the caching allocator in bytes for a given device. By default, this returns the peak cached memory since the beginning of this program. :func:`~torch.xpu.re... |
| `torch.xpu.mem_get_info` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `tuple[int, int]` | Return the global free and total GPU memory for a given device. Args: device (torch.device or int or str, optional): selected device. Returns statistic for the current device, given by :func:`~torc... |
| `torch.xpu.memory_allocated` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the current GPU memory occupied by tensors in bytes for a given device. Args: device (torch.device or int or str, optional): selected device. Returns statistic for the current device, given ... |
| `torch.xpu.memory_reserved` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the current GPU memory managed by the caching allocator in bytes for a given device. Args: device (torch.device or int or str, optional): selected device. Returns statistic for the current d... |
| `torch.xpu.memory_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `dict[str, typing.Any]` | Return a dictionary of XPU memory allocator statistics for a given device. The return value of this function is a dictionary of statistics, each of which is a non-negative integer. Core statistics:... |
| `torch.xpu.memory_stats_as_nested_dict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `dict[str, typing.Any]` | Return the result of :func:`~torch.xpu.memory_stats` as a nested dictionary. |
| `torch.xpu.reset_accumulated_memory_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Reset the "accumulated" (historical) stats tracked by the XPU memory allocator. See :func:`~torch.xpu.memory_stats` for details. Accumulated stats correspond to the `"allocated"` and `"freed"` keys... |
| `torch.xpu.reset_peak_memory_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Reset the "peak" stats tracked by the XPU memory allocator. See :func:`~torch.xpu.memory_stats` for details. Peak stats correspond to the `"peak"` key in each individual stat dict. Args: device (to... |
| `torch.xpu.seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Set the seed for generating random numbers to a random number for the current GPU. It's safe to call this function if XPU is not available; in that case, it is silently ignored. .. warning:: If you... |
| `torch.xpu.seed_all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Set the seed for generating random numbers to a random number on all GPUs. It's safe to call this function if XPU is not available; in that case, it is silently ignored. |
| `torch.xpu.set_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Set the current device. Args: device (torch.device or int or str): selected device. This function is a no-op if this argument is negative. |
| `torch.xpu.set_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `new_state, device` | `None` | Set the random number generator state of the specified GPU. Args: new_state (torch.ByteTensor): The desired state device (torch.device or int, optional): The device to set the RNG state. Default: `... |
| `torch.xpu.set_rng_state_all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `new_states` | `None` | Set the random number generator state of all devices. Args: new_states (Iterable of torch.ByteTensor): The desired state for each device. |
| `torch.xpu.set_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `Any` | Set the current stream.This is a wrapper API to set the stream. Usage of this function is discouraged in favor of the ``stream`` context manager. Args: stream (Stream): selected stream. This functi... |
| `torch.xpu.stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `<class 'torch.xpu.StreamContext'>` | Wrap around the Context-manager StreamContext that selects a given stream. Arguments: stream (Stream): selected stream. This manager is a no-op if it's ``None``. |
| `torch.xpu.synchronize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Wait for all kernels in all streams on a XPU device to complete. Args: device (torch.device or int, optional): device for which to synchronize. It uses the current device, given by :func:`~torch.xp... |

## Summary

- **Total Functions**: 1299
- **Completed**: 1
- **In Progress**: 0
- **Not Started**: 1298

## Implementation Status Tracking

### Overall Progress
- **Total Functions**: 1299
- **Completed**: 1
- **In Progress**: 0
- **Not Started**: 1298

### Priority Implementation Order
1. **Core Device Operations** (CUDA, MPS, CPU)
2. **Tensor Operations** (creation, manipulation, math)
3. **Neural Network Modules** (nn, functional)
4. **Optimization** (optim, autograd)
5. **Utilities** (utils, types, storage)

## Migration Strategy

### Priority Matrix

| Priority | Category | Functions | Rationale |
|----------|----------|-----------|-----------|
| **P0 (Critical)** | Device Management | 15 functions | Core functionality required |
| **P1 (High)** | Tensor Creation | 20 functions | Basic tensor operations |
| **P2 (Medium)** | Neural Network | 25 functions | ML model support |
| **P3 (Low)** | Mathematical | 30 functions | Advanced operations |
| **P4 (Nice-to-have)** | Utility | 40 functions | Helper functions |

### Implementation Phases

#### Phase 1: Foundation (Weeks 1-2)
- Device management functions
- Basic tensor creation
- Core mathematical operations

#### Phase 2: Neural Networks (Weeks 3-4)
- Activation functions
- Loss functions
- Basic neural network operations

#### Phase 3: Advanced Operations (Weeks 5-6)
- Complex mathematical operations
- Linear algebra functions
- Signal processing

#### Phase 4: Optimization (Weeks 7-8)
- Performance optimization
- Memory management
- Stream operations

## Architecture Overview

### Function Router Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PyTorch Call  │───▶│  TorchDevice    │───▶│  Device Router  │
│                 │    │  Interceptor    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CPU Device    │◀───│  Translation    │◀───│  Compatibility  │
│   Operations    │    │  Engine         │    │  Matrix         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CUDA Device   │◀───│  Fallback       │◀───│  Error Handler  │
│   Operations    │    │  Manager        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MPS Device    │◀───│  Performance    │◀───│  Monitoring     │
│   Operations    │    │  Optimizer      │    │  System         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

1. **Function Interceptor**: Hooks into PyTorch calls
2. **Device Router**: Determines optimal device for operation
3. **Translation Engine**: Converts between device types
4. **Compatibility Matrix**: Stores device support information
5. **Fallback Manager**: Handles unsupported operations
6. **Performance Optimizer**: Optimizes for specific devices
7. **Monitoring System**: Tracks performance and errors

### Success Metrics

- **Function Coverage**: 95% of PyTorch functions supported
- **Performance**: <5% overhead compared to native PyTorch
- **Compatibility**: 100% API compatibility with PyTorch
- **Error Rate**: <1% translation errors
- **Memory Usage**: <10% additional memory overhead

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API Changes | Medium | High | Version pinning, compatibility tests |
| Performance Degradation | High | Medium | Profiling, optimization |
| Memory Leaks | Low | High | Memory monitoring, cleanup |
| Device Compatibility | Medium | High | Comprehensive testing |

### Implementation Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Scope Creep | High | Medium | Phased approach, clear priorities |
| Testing Complexity | Medium | High | Automated testing, CI/CD |
| Documentation | Low | Medium | Auto-generated docs |
| Maintenance | Medium | Medium | Modular design, clear interfaces |

## Next Steps

1. Review function status and update implementation priorities
2. Implement core device functions (Phase 1)
3. Test with real-world projects
4. Iterate and optimize based on feedback

---
*Generated on 2025-07-10 15:47:21*
