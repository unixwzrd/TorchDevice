# Device Translation Implementation Guide
*Generated from PyTorch Function Analysis*

## Critical Functions by Category

### 1. Device Creation Functions (Phase 1 - HIGHEST PRIORITY)

#### torch.DeviceObjType
- **Category**: TORCH_DEVICE
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.device
- **Category**: TORCH_DEVICE
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.get_default_device
- **Category**: TORCH_DEVICE
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Gets the default ``torch.Tensor`` to be allocated on ``device``...

#### torch.get_device
- **Category**: TORCH_DEVICE
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.profiler_allow_cudagraph_cupti_lazy_reinit_cuda12
- **Category**: TORCH_DEVICE
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.set_default_device
- **Category**: TORCH_DEVICE
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Sets the default ``torch.Tensor`` to be allocated on ``device``.  This
does not affect factory function calls which are called with an explicit
``device`` argument.  Factory calls will be performed as...

#### torch._inductor.cudagraph_mark_step_begin
- **Category**: CORE_TORCH
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Indicates that a new iteration of inference or training is about to begin....

#### torch._lazy.wait_device_ops
- **Category**: CORE_TORCH
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Waits for all the async operations on the given devices to complete.
Args:
  devices (string..., optional): The devices whose async ops need to be waited
    for. If empty, all the local devices will ...

#### torch._prims_common.CUDARngStateHelper
- **Category**: CORE_TORCH
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch._prims_common.canonicalize_device
- **Category**: CORE_TORCH
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch._prims_common.check_same_device
- **Category**: CORE_TORCH
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Checks that all Tensors in args have the same device.

Raises a RuntimeError when:
  - args contains an object whose type is not Tensor or Number
  - two Tensor objects in args have different devices,...

#### torch._prims_common.device_or_default
- **Category**: CORE_TORCH
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.accelerator.current_device_idx
- **Category**: ACCELERATOR_SUPPORT
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the index of a currently selected device for the current :ref:`accelerator<accelerators>`.

Returns:
    int: the index of a currently selected device....

#### torch.accelerator.current_device_index
- **Category**: ACCELERATOR_SUPPORT
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the index of a currently selected device for the current :ref:`accelerator<accelerators>`.

Returns:
    int: the index of a currently selected device....

#### torch.accelerator.device_count
- **Category**: ACCELERATOR_SUPPORT
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the number of current :ref:`accelerator<accelerators>` available.

Returns:
    int: the number of the current :ref:`accelerator<accelerators>` available.
        If there is no available accel...

#### torch.accelerator.set_device_idx
- **Category**: ACCELERATOR_SUPPORT
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the current device index to a given device.

Args:
    device (:class:`torch.device`, str, int): a given device that must match the current
        :ref:`accelerator<accelerators>` device type.

....

#### torch.accelerator.set_device_index
- **Category**: ACCELERATOR_SUPPORT
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the current device index to a given device.

Args:
    device (:class:`torch.device`, str, int): a given device that must match the current
        :ref:`accelerator<accelerators>` device type.

....

#### torch.autograd.DeviceType
- **Category**: AUTOGRAD
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Members:

CPU

CUDA

MKLDNN

OPENGL

OPENCL

IDEEP

HIP

FPGA

MAIA

XLA

Vulkan

Metal

XPU

MPS

MTIA

Meta

HPU

VE

Lazy

IPU

PrivateUse1...

#### torch.compiler.cudagraph_mark_step_begin
- **Category**: COMPILATION
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Indicates that a new iteration of inference or training is about to begin.

CUDA Graphs will free tensors of a prior iteration. A new iteration is started on each invocation of
torch.compile, so long ...

#### torch.cpu.current_device
- **Category**: CPU_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Returns current device for cpu. Always 'cpu'.

N.B. This function only exists to facilitate device-agnostic code...

#### torch.cpu.device_count
- **Category**: CPU_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Returns number of CPU devices (not cores). Always 1.

N.B. This function only exists to facilitate device-agnostic code...

#### torch.cpu.set_device
- **Category**: CPU_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Sets the current device, in CPU we do nothing.

N.B. This function only exists to facilitate device-agnostic code...

#### torch.cuda.Any
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements...

#### torch.cuda.BFloat16Storage
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.BFloat16Tensor
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.BoolStorage
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.BoolTensor
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.ByteStorage
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.ByteTensor
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.CUDAGraph
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Wrapper around a CUDA graph.

.. warning::
    This API is in beta and may change in future releases....

#### torch.cuda.CUDAPluggableAllocator
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: CUDA memory allocator loaded from a so file....

#### torch.cuda.CharStorage
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.CharTensor
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.ComplexDoubleStorage
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.ComplexFloatStorage
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.CudaError
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Unspecified run-time error....

#### torch.cuda.DeferredCudaCallError
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Common base class for all non-exit exceptions....

#### torch.cuda.DoubleStorage
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.DoubleTensor
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.Event
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Wrapper around a CUDA event.

CUDA events are synchronization markers that can be used to monitor the
device's progress, to accurately measure timing, and to synchronize CUDA
streams.

The underlying ...

#### torch.cuda.ExternalStream
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Wrapper around an externally allocated CUDA stream.

This class is used to wrap streams allocated in other libraries in order
to facilitate data exchange and multi-library interactions.

.. note:: Thi...

#### torch.cuda.FloatStorage
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.FloatTensor
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.HalfStorage
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.HalfTensor
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.IntStorage
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.IntTensor
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.LongStorage
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.LongTensor
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.MemPool
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: MemPool represents a pool of memory in a caching allocator. Currently,
it's just the ID of the pool object maintained in the CUDACachingAllocator.

Args:
    allocator(torch._C._cuda_CUDAAllocator, op...

#### torch.cuda.MemPoolContext
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: MemPoolContext holds the currently active pool and stashes the previous
pool. On deletion it makes the previous pool active.

Args:
    pool(torch.cuda.MemPool): a MemPool object to be made active so ...

#### torch.cuda.OutOfMemoryError
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Exception raised when device is out of memory...

#### torch.cuda.ShortStorage
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.ShortTensor
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.Stream
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Wrapper around a CUDA stream.

A CUDA stream is a linear sequence of execution that belongs to a specific
device, independent from other streams. It supports with statement as a
context manager to ens...

#### torch.cuda.StreamContext
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Context-manager that selects a given stream.

All CUDA kernels queued within its context will be enqueued on a selected
stream.

Args:
    Stream (Stream): selected stream. This manager is a no-op if ...

#### torch.cuda.caching_allocator_alloc
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Perform a memory allocation using the CUDA memory allocator.

Memory is allocated for a given device and a stream, this
function is intended to be used for interoperability with other
frameworks. Allo...

#### torch.cuda.caching_allocator_delete
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Delete memory allocated using the CUDA memory allocator.

Memory allocated with :func:`~torch.cuda.caching_allocator_alloc`.
is freed here. The associated device and stream are tracked inside
the allo...

#### torch.cuda.caching_allocator_enable
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Enable or disable the CUDA memory allocator. On by default....

#### torch.cuda.can_device_access_peer
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Check if peer access between two devices is possible....

#### torch.cuda.cast
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Cast a value to a type.

This returns the value unchanged.  To the type checker this
signals that the return value has the designated type, but at
runtime we intentionally don't check anything (we wan...

#### torch.cuda.change_current_allocator
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Change the currently used memory allocator to be the one provided.

If the current allocator has already been used/initialized, this function will error.


Args:
    allocator (torch.cuda.memory._CUDA...

#### torch.cuda.check_error
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.classproperty
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.clock_rate
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the clock speed of the GPU SM in MHz (megahertz) over the past sample period as given by `nvidia-smi`.

Args:
    device (torch.device or int, optional): selected device. Returns
        statis...

#### torch.cuda.cudaStatus
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.cuda.cudart
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Retrieves the CUDA runtime API module.


This function initializes the CUDA runtime environment if it is not already
initialized and returns the CUDA runtime API module (_cudart). The CUDA
runtime API...

#### torch.cuda.current_blas_handle
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return cublasHandle_t pointer to current cuBLAS handle...

#### torch.cuda.current_device
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the index of a currently selected device....

#### torch.cuda.current_stream
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the currently selected :class:`Stream` for a given device.

Args:
    device (torch.device or int, optional): selected device. Returns
        the currently selected :class:`Stream` for the cur...

#### torch.cuda.default_stream
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the default :class:`Stream` for a given device.

Args:
    device (torch.device or int, optional): selected device. Returns
        the default :class:`Stream` for the current device, given by
...

#### torch.cuda.device
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Context-manager that changes the selected device.

Args:
    device (torch.device or int): device index to select. It's a no-op if
        this argument is a negative integer or ``None``....

#### torch.cuda.device_count
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the number of GPUs available.

.. note:: This API will NOT posion fork if NVML discovery succeeds.
    See :ref:`multiprocessing-poison-fork-note` for more details....

#### torch.cuda.device_memory_used
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return used global (device) memory in bytes as given by `nvidia-smi` or `amd-smi`.

Args:
    device (torch.device or int, optional): selected device. Returns
        statistic for the current device,...

#### torch.cuda.device_of
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Context-manager that changes the current device to that of given object.

You can use both tensors and storages as arguments. If a given object is
not allocated on a GPU, this is a no-op.

Args:
    o...

#### torch.cuda.empty_cache
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Release all unoccupied cached memory currently held by the caching
allocator so that those can be used in other GPU application and visible in
`nvidia-smi`.

.. note::
    :func:`~torch.cuda.empty_cac...

#### torch.cuda.get_allocator_backend
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a string describing the active allocator backend as set by
``PYTORCH_CUDA_ALLOC_CONF``. Currently available backends are
``native`` (PyTorch's native caching allocator) and `cudaMallocAsync``
(...

#### torch.cuda.get_arch_list
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return list CUDA architectures this library was compiled for....

#### torch.cuda.get_device_capability
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Get the cuda capability of a device.

Args:
    device (torch.device or int or str, optional): device for which to return the
        device capability. This function is a no-op if this argument is
  ...

#### torch.cuda.get_device_name
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Get the name of a device.

Args:
    device (torch.device or int or str, optional): device for which to return the
        name. This function is a no-op if this argument is a negative
        integer...

#### torch.cuda.get_device_properties
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Get the properties of a device.

Args:
    device (torch.device or int or str, optional): device for which to return the
        properties of the device.  It uses the current device, given by
       ...

#### torch.cuda.get_gencode_flags
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return NVCC gencode flags this library was compiled with....

#### torch.cuda.get_per_process_memory_fraction
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Get memory fraction for a process.

Args:
    device (torch.device or int, optional): selected device. If it is
        ``None`` the default CUDA device is used.
Returns:
    memory fraction, in range...

#### torch.cuda.get_rng_state
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the random number generator state of the specified GPU as a ByteTensor.

Args:
    device (torch.device or int, optional): The device to return the RNG state of.
        Default: ``'cuda'`` (i....

#### torch.cuda.get_rng_state_all
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a list of ByteTensor representing the random number states of all devices....

#### torch.cuda.get_stream_from_external
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a :class:`Stream` from an externally allocated CUDA stream.

This function is used to wrap streams allocated in other libraries in order
to facilitate data exchange and multi-library interactio...

#### torch.cuda.get_sync_debug_mode
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return current value of debug mode for cuda synchronizing operations....

#### torch.cuda.graph
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Context-manager that captures CUDA work into a :class:`torch.cuda.CUDAGraph` object for later replay.

See :ref:`CUDA Graphs <cuda-graph-semantics>` for a general introduction,
detailed use, and const...

#### torch.cuda.graph_pool_handle
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return an opaque token representing the id of a graph memory pool.

See :ref:`Graph memory management<graph-memory-management>`.

.. warning::
    This API is in beta and may change in future releases...

#### torch.cuda.host_memory_stats
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a dictionary of CUDA memory allocator statistics for a given device.

 The return value of this function is a dictionary of statistics, each of
 which is a non-negative integer.

 Core statisti...

#### torch.cuda.host_memory_stats_as_nested_dict
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the result of :func:`~torch.cuda.host_memory_stats` as a nested dictionary....

#### torch.cuda.init
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Initialize PyTorch's CUDA state.

You may need to call this explicitly if you are interacting with
PyTorch via its C API, as Python bindings for CUDA functionality
will not be available until this ini...

#### torch.cuda.initial_seed
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the current random seed of the current GPU.

.. warning::
    This function eagerly initializes CUDA....

#### torch.cuda.ipc_collect
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Force collects GPU memory after it has been released by CUDA IPC.

.. note::
    Checks if any sent CUDA tensors could be cleaned from the memory. Force
    closes shared memory file used for referenc...

#### torch.cuda.is_available
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a bool indicating if CUDA is currently available.

.. note:: This function will NOT poison fork if the environment variable
    ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set. For more details, see...

#### torch.cuda.is_bf16_supported
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a bool indicating if the current CUDA/ROCm device supports dtype bfloat16....

#### torch.cuda.is_current_stream_capturing
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return True if CUDA graph capture is underway on the current CUDA stream, False otherwise.

If a CUDA context does not exist on the current device, returns False without initializing the context....

#### torch.cuda.is_initialized
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return whether PyTorch's CUDA state has been initialized....

#### torch.cuda.is_tf32_supported
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a bool indicating if the current CUDA/ROCm device supports dtype tf32....

#### torch.cuda.list_gpu_processes
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a human-readable printout of the running processes and their GPU memory use for a given device.

This can be useful to display periodically during training, or when
handling out-of-memory excep...

#### torch.cuda.lru_cache
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Least-recently-used cache decorator.

If *maxsize* is set to None, the LRU features are disabled and the cache
can grow without bound.

If *typed* is True, arguments of different types will be cached ...

#### torch.cuda.make_graphed_callables
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Accept callables (functions or :class:`nn.Module<torch.nn.Module>`\ s) and returns graphed versions.

Each graphed callable's forward pass runs its source callable's
forward CUDA work as a CUDA graph ...

#### torch.cuda.manual_seed
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the seed for generating random numbers for the current GPU.

It's safe to call this function if CUDA is not available; in that
case, it is silently ignored.

Args:
    seed (int): The desired seed...

#### torch.cuda.manual_seed_all
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the seed for generating random numbers on all GPUs.

It's safe to call this function if CUDA is not available; in that
case, it is silently ignored.

Args:
    seed (int): The desired seed....

#### torch.cuda.max_memory_allocated
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the maximum GPU memory occupied by tensors in bytes for a given device.

By default, this returns the peak allocated memory since the beginning of
this program. :func:`~torch.cuda.reset_peak_me...

#### torch.cuda.max_memory_cached
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Deprecated; see :func:`~torch.cuda.max_memory_reserved`....

#### torch.cuda.max_memory_reserved
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the maximum GPU memory managed by the caching allocator in bytes for a given device.

By default, this returns the peak cached memory since the beginning of this
program. :func:`~torch.cuda.res...

#### torch.cuda.mem_get_info
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the global free and total GPU memory for a given device using cudaMemGetInfo.

Args:
    device (torch.device or int or str, optional): selected device. Returns
        statistic for the curren...

#### torch.cuda.memory_allocated
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the current GPU memory occupied by tensors in bytes for a given device.

Args:
    device (torch.device or int, optional): selected device. Returns
        statistic for the current device, giv...

#### torch.cuda.memory_cached
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Deprecated; see :func:`~torch.cuda.memory_reserved`....

#### torch.cuda.memory_reserved
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the current GPU memory managed by the caching allocator in bytes for a given device.

Args:
    device (torch.device or int, optional): selected device. Returns
        statistic for the curren...

#### torch.cuda.memory_snapshot
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a snapshot of the CUDA memory allocator state across all devices.

Interpreting the output of this function requires familiarity with the
memory allocator internals.

.. note::
    See :ref:`cu...

#### torch.cuda.memory_stats
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a dictionary of CUDA memory allocator statistics for a given device.

The return value of this function is a dictionary of statistics, each of
which is a non-negative integer.

Core statistics:...

#### torch.cuda.memory_stats_as_nested_dict
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the result of :func:`~torch.cuda.memory_stats` as a nested dictionary....

#### torch.cuda.memory_summary
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return a human-readable printout of the current memory allocator statistics for a given device.

This can be useful to display periodically during training, or when
handling out-of-memory exceptions.
...

#### torch.cuda.memory_usage
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the percent of time over the past sample period during which global (device)
memory was being read or written as given by `nvidia-smi`.

Args:
    device (torch.device or int, optional): select...

#### torch.cuda.power_draw
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the average power draw of the GPU sensor in mW (MilliWatts)
    over the past sample period as given by `nvidia-smi` for Fermi or newer fully supported devices.

Args:
    device (torch.device ...

#### torch.cuda.reset_accumulated_host_memory_stats
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Reset the "accumulated" (historical) stats tracked by the host memory allocator.

See :func:`~torch.cuda.host_memory_stats` for details. Accumulated stats correspond to
the `"allocated"` and `"freed"`...

#### torch.cuda.reset_accumulated_memory_stats
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Reset the "accumulated" (historical) stats tracked by the CUDA memory allocator.

See :func:`~torch.cuda.memory_stats` for details. Accumulated stats correspond to
the `"allocated"` and `"freed"` keys...

#### torch.cuda.reset_max_memory_allocated
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Reset the starting point in tracking maximum GPU memory occupied by tensors for a given device.

See :func:`~torch.cuda.max_memory_allocated` for details.

Args:
    device (torch.device or int, optio...

#### torch.cuda.reset_max_memory_cached
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Reset the starting point in tracking maximum GPU memory managed by the caching allocator for a given device.

See :func:`~torch.cuda.max_memory_cached` for details.

Args:
    device (torch.device or ...

#### torch.cuda.reset_peak_host_memory_stats
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Reset the "peak" stats tracked by the host memory allocator.

See :func:`~torch.cuda.host_memory_stats` for details. Peak stats correspond to the
`"peak"` key in each individual stat dict....

#### torch.cuda.reset_peak_memory_stats
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Reset the "peak" stats tracked by the CUDA memory allocator.

See :func:`~torch.cuda.memory_stats` for details. Peak stats correspond to the
`"peak"` key in each individual stat dict.

Args:
    devic...

#### torch.cuda.seed
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the seed for generating random numbers to a random number for the current GPU.

It's safe to call this function if CUDA is not available; in that
case, it is silently ignored.

.. warning::
    If...

#### torch.cuda.seed_all
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the seed for generating random numbers to a random number on all GPUs.

It's safe to call this function if CUDA is not available; in that
case, it is silently ignored....

#### torch.cuda.set_device
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the current device.

Usage of this function is discouraged in favor of :any:`device`. In most
cases it's better to use ``CUDA_VISIBLE_DEVICES`` environmental variable.

Args:
    device (torch.dev...

#### torch.cuda.set_per_process_memory_fraction
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set memory fraction for a process.

The fraction is used to limit an caching allocator to allocated memory on a CUDA device.
The allowed value equals the total visible memory multiplied fraction.
If t...

#### torch.cuda.set_rng_state
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the random number generator state of the specified GPU.

Args:
    new_state (torch.ByteTensor): The desired state
    device (torch.device or int, optional): The device to set the RNG state.
    ...

#### torch.cuda.set_rng_state_all
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the random number generator state of all devices.

Args:
    new_states (Iterable of torch.ByteTensor): The desired state for each device....

#### torch.cuda.set_stream
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the current stream.This is a wrapper API to set the stream.
    Usage of this function is discouraged in favor of the ``stream``
    context manager.

Args:
    stream (Stream): selected stream. T...

#### torch.cuda.set_sync_debug_mode
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the debug mode for cuda synchronizing operations.

Args:
    debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations,
        if "warn" or 1, warn on synchronizi...

#### torch.cuda.stream
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Wrap around the Context-manager StreamContext that selects a given stream.

Arguments:
    stream (Stream): selected stream. This manager is a no-op if it's
        ``None``.
.. note::
    In eager mo...

#### torch.cuda.synchronize
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Wait for all kernels in all streams on a CUDA device to complete.

Args:
    device (torch.device or int, optional): device for which to synchronize.
        It uses the current device, given by :func...

#### torch.cuda.temperature
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the average temperature of the GPU sensor in Degrees C (Centigrades).

The average temperature is computed based on past sample period as given by `nvidia-smi`.

Args:
    device (torch.device ...

#### torch.cuda.use_mem_pool
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: A context manager that routes allocations to a given pool.

Args:
    pool(torch.cuda.MemPool): a MemPool object to be made active so that
        allocations route to this pool.
    device (torch.dev...

#### torch.cuda.utilization
- **Category**: CUDA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the percent of time over the past sample period during which one or
more kernels was executing on the GPU as given by `nvidia-smi`.

Args:
    device (torch.device or int, optional): selected d...

#### torch.distributed.DeviceMesh
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: DeviceMesh represents a mesh of devices, where layout of devices could be
represented as a n-d dimension array, and each value of the n-d dimensional
array is the global id of the default process grou...

#### torch.distributed.get_default_backend_for_device
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the default backend for the given device.

Args:
    Union[str, torch.device]: The device to get the default backend for.

Returns:
    The default backend for the given device as a lower case ...

#### torch.distributed.init_device_mesh
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Initializes a `DeviceMesh` based on `device_type`, `mesh_shape`, and `mesh_dim_names` parameters.

This creates a DeviceMesh with an n-dimensional array layout, where `n` is the length of `mesh_shape`...

#### torch.mps.device_count
- **Category**: MPS_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Returns the number of available MPS devices....

#### torch.mtia.current_device
- **Category**: MTIA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the index of a currently selected device....

#### torch.mtia.device
- **Category**: MTIA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Context-manager that changes the selected device.

Args:
    device (torch.device or int): device index to select. It's a no-op if
        this argument is a negative integer or ``None``....

#### torch.mtia.device_count
- **Category**: MTIA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the number of MTIA devices available....

#### torch.mtia.get_device_capability
- **Category**: MTIA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return capability of a given device as a tuple of (major version, minor version).

Args:
    device (torch.device or int, optional) selected device. Returns
        statistics for the current device, ...

#### torch.mtia.set_device
- **Category**: MTIA_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the current device.

Args:
    device (torch.device or int): selected device. This function is a no-op
        if this argument is negative....

#### torch.nested.Device
- **Category**: NESTED_TENSORS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: ...

#### torch.profiler.DeviceType
- **Category**: PROFILING
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Members:

CPU

CUDA

MKLDNN

OPENGL

OPENCL

IDEEP

HIP

FPGA

MAIA

XLA

Vulkan

Metal

XPU

MPS

MTIA

Meta

HPU

VE

Lazy

IPU

PrivateUse1...

#### torch.xpu.current_device
- **Category**: XPU_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Return the index of a currently selected device....

#### torch.xpu.device
- **Category**: XPU_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Context-manager that changes the selected device.

Args:
    device (torch.device or int or str): device index to select. It's a no-op if
        this argument is a negative integer or ``None``....

#### torch.xpu.device_of
- **Category**: XPU_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Context-manager that changes the current device to that of given object.

You can use both tensors and storages as arguments. If a given object is
not allocated on a XPU, this is a no-op.

Args:
    o...

#### torch.xpu.get_device_name
- **Category**: XPU_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Get the name of a device.

Args:
    device (torch.device or int or str, optional): device for which to
        return the name. This function is a no-op if this argument is a
        negative integer...

#### torch.xpu.get_device_properties
- **Category**: XPU_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Get the properties of a device.

Args:
    device (torch.device or int or str): device for which to return the
        properties of the device.

Returns:
    _XpuDeviceProperties: the properties of t...

#### torch.xpu.set_device
- **Category**: XPU_OPERATIONS
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: Set the current device.

Args:
    device (torch.device or int or str): selected device. This function is a
        no-op if this argument is negative....

### 2. Tensor Creation Functions (Phase 1 - HIGHEST PRIORITY)

#### torch.BFloat16Tensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.BoolTensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.ByteTensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.CharTensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.DoubleTensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.FloatTensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.HalfTensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.IntTensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.LongTensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.ShortTensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.Tensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.TensorType
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.align_tensors
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.as_tensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: as_tensor(data: Any, dtype: Optional[dtype] = None, device: Optional[DeviceLikeType]) -> Tensor

Converts :attr:`data` into a tensor, sharing data and preserving autograd
history if possible.

If :att...

#### torch.broadcast_tensors
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: broadcast_tensors(*tensors) -> List of Tensors

Broadcasts the given tensors according to :ref:`broadcasting-semantics`.

Args:
    *tensors: any number of tensors of the same type

.. warning::

    ...

#### torch.fake_quantize_per_tensor_affine
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max) -> Tensor

Returns a new tensor with the data in :attr:`input` fake quantized using :attr:`scale`,
:attr:`zero_point`, :...

#### torch.is_tensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Returns True if `obj` is a PyTorch tensor.

Note that this function is simply doing ``isinstance(obj, Tensor)``.
Using that ``isinstance`` check is better for typechecking with mypy,
and more explicit...

#### torch.quantize_per_tensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: quantize_per_tensor(input, scale, zero_point, dtype) -> Tensor

Converts a float tensor to a quantized tensor with given scale and zero point.

Arguments:
    input (Tensor): float tensor or list of t...

#### torch.quantize_per_tensor_dynamic
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: quantize_per_tensor_dynamic(input, dtype, reduce_range) -> Tensor

Converts a float tensor to a quantized tensor with scale and zero_point calculated
dynamically based on the input.

Arguments:
    in...

#### torch.scalar_tensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.set_default_tensor_type
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: .. warning::

    This function is deprecated as of PyTorch 2.1, please use :func:`torch.set_default_dtype()` and
    :func:`torch.set_default_device()` as alternatives.

Sets the default ``torch.Tens...

#### torch.sparse_bsc_tensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: sparse_bsc_tensor(ccol_indices, row_indices, values, size=None, *, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None) -> Tensor

Constructs a :ref:`sparse tensor in...

#### torch.sparse_bsr_tensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: sparse_bsr_tensor(crow_indices, col_indices, values, size=None, *, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None) -> Tensor

Constructs a :ref:`sparse tensor in...

#### torch.sparse_compressed_tensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: sparse_compressed_tensor(compressed_indices, plain_indices, values, size=None, *, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None) -> Tensor

Constru...

#### torch.sparse_coo_tensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: sparse_coo_tensor(indices, values, size=None, *, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None, is_coalesced=None) -> Tensor

Constructs a :ref:`sparse tensor i...

#### torch.sparse_csc_tensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: sparse_csc_tensor(ccol_indices, row_indices, values, size=None, *, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None) -> Tensor

Constructs a :ref:`sparse tensor in...

#### torch.sparse_csr_tensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: sparse_csr_tensor(crow_indices, col_indices, values, size=None, *, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None) -> Tensor

Constructs a :ref:`sparse tensor in...

#### torch.tensor
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor

Constructs a tensor with no autograd history (also known as a "leaf tensor", see :doc:`/notes/autograd`) by c...

#### torch.tensor_split
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: tensor_split(input, indices_or_sections, dim=0) -> List of Tensors

Splits a tensor into multiple sub-tensors, all of which are views of :attr:`input`,
along dimension :attr:`dim` according to the ind...

#### torch.tensordot
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Returns a contraction of a and b over multiple dimensions.

:attr:`tensordot` implements a generalized matrix product.

Args:
  a (Tensor): Left tensor to contract
  b (Tensor): Right tensor to contra...

#### torch._decomp.FunctionalTensor
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Functional tensors represent tensors that will remove mutations
from a program. If you perform a mutable operation on a functional tensor,
it will re-dispatch to the functional variant of that operati...

#### torch._dynamo.TensorifyState
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._export.TensorArgument
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: TensorArgument(name: str)...

#### torch._lazy.get_tensor_id
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Return a unique id of the lazy tensor maintained by LTC...

#### torch._numpy.tensordot
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._prims.FakeTensor
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Meta tensors give you the ability to run PyTorch code without having to
actually do computation through tensors allocated on a `meta` device.
Because the device is `meta`, meta tensors do not model de...

#### torch._prims.FakeTensorMode
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: A ``TorchDispatchMode`` allows you to override the meaning of all
``__torch_dispatch__`` overrideable functions within a dynamic scope,
without having to actually create a tensor subclass or manually
...

#### torch._prims.Tensor
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._prims.TensorLike
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._prims.TensorLikeType
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._prims.TensorMeta
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._prims.new_token_tensor
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._prims_common.Tensor
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._prims_common.TensorLike
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._prims_common.TensorLikeType
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._prims_common.compare_tensor_meta
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Checks that two tensor likes have the same shape,
dtype and device.

In the future this will validate additional metadata, like
strides....

#### torch._prims_common.is_cpu_scalar_tensor
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._prims_common.mask_tensor
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Similar to torch.where(mask, t, 0) but if t is boolean,
result is also boolean and not promoted to int....

#### torch._refs.Tensor
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._refs.TensorLike
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._refs.TensorLikeType
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._refs.broadcast_tensors
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._refs.scalar_tensor
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._refs.tensor
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._refs.tensor_split
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch._subclasses.FakeTensor
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Meta tensors give you the ability to run PyTorch code without having to
actually do computation through tensors allocated on a `meta` device.
Because the device is `meta`, meta tensors do not model de...

#### torch._subclasses.FakeTensorMode
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: A ``TorchDispatchMode`` allows you to override the meaning of all
``__torch_dispatch__`` overrideable functions within a dynamic scope,
without having to actually create a tensor subclass or manually
...

#### torch._subclasses.UnsupportedFakeTensorException
- **Category**: CORE_TORCH
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: UnsupportedFakeTensorException(reason: 'str')...

#### torch.autograd.SavedTensor
- **Category**: AUTOGRAD
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.autograd.is_tensor_like
- **Category**: AUTOGRAD
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Returns ``True`` if the passed-in input is a Tensor-like.

Currently, this occurs whenever there's a ``__torch_function__``
attribute on the type of the input.

Examples
--------
A subclass of tensor ...

#### torch.distributed.all_gather_into_tensor
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Gather tensors from all ranks and put them in a single output tensor.

This function requires all tensors to be the same size on each process.

Args:
    output_tensor (Tensor): Output tensor to accom...

#### torch.distributed.reduce_scatter_tensor
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Reduces, then scatters a tensor to all ranks in a group.

Args:
    output (Tensor): Output tensor. It should have the same size across all
        ranks.
    input (Tensor): Input tensor to be reduce...

#### torch.fft.Tensor
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.linalg.tensorinv
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: linalg.tensorinv(A, ind=2, *, out=None) -> Tensor

Computes the multiplicative inverse of :func:`torch.tensordot`.

If `m` is the product of the first :attr:`ind` dimensions of :attr:`A` and `n` is th...

#### torch.linalg.tensorsolve
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: linalg.tensorsolve(A, B, dims=None, *, out=None) -> Tensor

Computes the solution `X` to the system `torch.tensordot(A, X) = B`.

If `m` is the product of the first :attr:`B`\ `.ndim`  dimensions of :...

#### torch.masked.MaskedTensor
- **Category**: MASKED_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.masked.as_masked_tensor
- **Category**: MASKED_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.masked.is_masked_tensor
- **Category**: MASKED_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Returns True if the input is a MaskedTensor, else False

Args:
    a: any input

Examples:

    >>> # xdoctest: +SKIP
    >>> from torch.masked import MaskedTensor
    >>> data = torch.arange(6).resha...

#### torch.masked.masked_tensor
- **Category**: MASKED_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.monitor.TensorboardEventHandler
- **Category**: MONITORING
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: TensorboardEventHandler is an event handler that will write known events to
the provided SummaryWriter.

This currently only supports ``torch.monitor.Stat`` events which are logged
as scalars.

Exampl...

#### torch.mps.Tensor
- **Category**: MPS_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.mtia.Tensor
- **Category**: MTIA_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.nested.Tensor
- **Category**: NESTED_TENSORS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.nested.as_nested_tensor
- **Category**: NESTED_TENSORS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Constructs a nested tensor preserving autograd history from a tensor or a list / tuple of
tensors.

If a nested tensor is passed, it will be returned directly unless the device / dtype / layout
differ...

#### torch.nested.nested_tensor
- **Category**: NESTED_TENSORS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Constructs a nested tensor with no autograd history (also known as a "leaf tensor", see
:ref:`Autograd mechanics <autograd-mechanics>`) from :attr:`tensor_list` a list of tensors.

Args:
    tensor_li...

#### torch.nested.nested_tensor_from_jagged
- **Category**: NESTED_TENSORS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Constructs a jagged layout nested tensor from the given jagged components. The jagged layout
consists of a required values buffer with the jagged dimension packed into a single dimension.
The offsets ...

#### torch.nested.to_padded_tensor
- **Category**: NESTED_TENSORS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: to_padded_tensor(input, padding, output_size=None, out=None) -> Tensor

Returns a new (non-nested) Tensor by padding the :attr:`input` nested tensor.
The leading entries will be filled with the nested...

#### torch.nn.functional.Tensor
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.onnx.TensorProtoDataType
- **Category**: ONNX_EXPORT
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Members:

UNDEFINED

FLOAT

UINT8

INT8

UINT16

INT16

INT32

INT64

STRING

BOOL

FLOAT16

DOUBLE

UINT32

UINT64

COMPLEX64

COMPLEX128

BFLOAT16

FLOAT8E4M3FN

FLOAT8E4M3FNUZ

FLOAT8E5M2

FLOAT8E5...

#### torch.overrides.is_tensor_like
- **Category**: CUSTOMIZATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Returns ``True`` if the passed-in input is a Tensor-like.

Currently, this occurs whenever there's a ``__torch_function__``
attribute on the type of the input.

Examples
--------
A subclass of tensor ...

#### torch.overrides.is_tensor_method_or_property
- **Category**: CUSTOMIZATION
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Returns True if the function passed in is a handler for a
method or property belonging to ``torch.Tensor``, as passed
into ``__torch_function__``.

.. note::
   For properties, their ``__get__`` metho...

#### torch.profiler.tensorboard_trace_handler
- **Category**: PROFILING
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Outputs tracing files to directory of ``dir_name``, then that directory can be
directly delivered to tensorboard as logdir.
``worker_name`` should be unique for each worker in distributed scenario,
it...

#### torch.sparse.BFloat16Tensor
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.sparse.ByteTensor
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.sparse.CharTensor
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.sparse.DoubleTensor
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.sparse.FloatTensor
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.sparse.HalfTensor
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.sparse.IntTensor
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.sparse.LongTensor
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.sparse.ShortTensor
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.sparse.SparseSemiStructuredTensor
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: This class implementes semi-structured sparsity as a Tensor subclass.

Semi-structured sparsity describes a sparsity pattern where n in every 2n elements are sparse,
depending on the datatype. It is a...

#### torch.sparse.SparseSemiStructuredTensorCUSPARSELT
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: The cuSPARSELt backend expects the specified elements and the metadata to be stored in a single tensor:
packed = [ specified elements of original tensor | metadata ]
For an original tensor of size (m,...

#### torch.sparse.SparseSemiStructuredTensorCUTLASS
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: This class implements semi-structured sparsity for the CUTLASS backend.


In this implementation, the specified elements and metadata are stored seprately,
in packed and meta respectively.

When _FORC...

#### torch.sparse.Tensor
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.sparse.check_sparse_tensor_invariants
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: A tool to control checking sparse tensor invariants.

The following options exists to manage sparsr tensor invariants
checking in sparse tensor construction:

1. Using a context manager:

   .. code::...

#### torch.special.Tensor
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.testing.make_tensor
- **Category**: TESTING_UTILITIES
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: Creates a tensor with the given :attr:`shape`, :attr:`device`, and :attr:`dtype`, and filled with
values uniformly drawn from ``[low, high)``.

If :attr:`low` or :attr:`high` are specified and are out...

#### torch.types.Tensor
- **Category**: TYPE_SYSTEM
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: ...

#### torch.utils.swap_tensors
- **Category**: UTILITIES
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: This function swaps the content of the two Tensor objects.
At a high level, this will make t1 have the content of t2 while preserving
its identity.

This will not work if t1 and t2 have different slot...

### 3. Device Management Functions (Phase 2 - MEDIUM PRIORITY)

### 4. Events Functions (Phase 2 - MEDIUM PRIORITY)

#### torch.Event
- **Category**: TORCH_EVENTS
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: Event(device, *, enable_timing) -> Event

Query and record Stream status to identify or control dependencies across Stream and measure timing.

Arguments:
    device (:class:`torch.device`, optional):...

#### torch.autograd.ProfilerEvent
- **Category**: AUTOGRAD
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: ...

#### torch.cpu.Event
- **Category**: CPU_OPERATIONS
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: ...

#### torch.monitor.Event
- **Category**: MONITORING
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: Event represents a specific typed event to be logged. This can represent
high-level data points such as loss or accuracy per epoch or more
low-level aggregations such as through the Stats provided thr...

#### torch.monitor.EventHandlerHandle
- **Category**: MONITORING
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: EventHandlerHandle is a wrapper type returned by
``register_event_handler`` used to unregister the handler via
``unregister_event_handler``. This cannot be directly initialized....

#### torch.monitor.log_event
- **Category**: MONITORING
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: log_event(event: torch._C._monitor.Event) -> None


log_event logs the specified event to all of the registered event
handlers. It's up to the event handlers to log the event out to the
corresponding ...

#### torch.monitor.register_event_handler
- **Category**: MONITORING
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: register_event_handler(callback: Callable[[torch._C._monitor.Event], None]) -> torch._C._monitor.EventHandlerHandle


register_event_handler registers a callback to be called whenever an
event is logg...

#### torch.monitor.unregister_event_handler
- **Category**: MONITORING
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: unregister_event_handler(handler: torch._C._monitor.EventHandlerHandle) -> None


unregister_event_handler unregisters the ``EventHandlerHandle`` returned
after calling ``register_event_handler``. Aft...

#### torch.mps.Event
- **Category**: MPS_OPERATIONS
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: Wrapper around an MPS event.

MPS events are synchronization markers that can be used to monitor the
device's progress, to accurately measure timing, and to synchronize MPS streams.

Args:
    enable_...

#### torch.mtia.Event
- **Category**: MTIA_OPERATIONS
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: Event(device, *, enable_timing) -> Event

Query and record Stream status to identify or control dependencies across Stream and measure timing.

Arguments:
    device (:class:`torch.device`, optional):...

#### torch.xpu.Event
- **Category**: XPU_OPERATIONS
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: Wrapper around a XPU event.

XPU events are synchronization markers that can be used to monitor the
device's progress, and to synchronize XPU streams.

The underlying XPU events are lazily initialized...

### 5. Streams Functions (Phase 2 - MEDIUM PRIORITY)

#### torch.Stream
- **Category**: TORCH_STREAMS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Stream(device, *, priority) -> Stream

An in-order queue of executing the respective tasks asynchronously in first in first out (FIFO) order.
It can control or synchronize the execution of other Strea...

#### torch.StreamObjType
- **Category**: TORCH_STREAMS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: ...

#### torch.accelerator.current_stream
- **Category**: ACCELERATOR_SUPPORT
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Return the currently selected stream for a given device.

Args:
    device (:class:`torch.device`, str, int, optional): a given device that must match the current
        :ref:`accelerator<accelerator...

#### torch.accelerator.set_stream
- **Category**: ACCELERATOR_SUPPORT
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Set the current stream to a given stream.

Args:
    stream (torch.Stream): a given stream that must match the current :ref:`accelerator<accelerators>` device type.

.. note:: This function will set t...

#### torch.cpu.Stream
- **Category**: CPU_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: N.B. This class only exists to facilitate device-agnostic code...

#### torch.cpu.StreamContext
- **Category**: CPU_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Context-manager that selects a given stream.

N.B. This class only exists to facilitate device-agnostic code...

#### torch.cpu.current_stream
- **Category**: CPU_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Returns the currently selected :class:`Stream` for a given device.

Args:
    device (torch.device or int, optional): Ignored.

N.B. This function only exists to facilitate device-agnostic code...

#### torch.cpu.stream
- **Category**: CPU_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Wrapper around the Context-manager StreamContext that
selects a given stream.

N.B. This function only exists to facilitate device-agnostic code...

#### torch.mtia.Stream
- **Category**: MTIA_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Stream(device, *, priority) -> Stream

An in-order queue of executing the respective tasks asynchronously in first in first out (FIFO) order.
It can control or synchronize the execution of other Strea...

#### torch.mtia.StreamContext
- **Category**: MTIA_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Context-manager that selects a given stream.

All MTIA kernels queued within its context will be enqueued on a selected
stream.

Args:
    Stream (Stream): selected stream. This manager is a no-op if ...

#### torch.mtia.current_stream
- **Category**: MTIA_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Return the currently selected :class:`Stream` for a given device.

Args:
    device (torch.device or int, optional): selected device. Returns
        the currently selected :class:`Stream` for the cur...

#### torch.mtia.default_stream
- **Category**: MTIA_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Return the default :class:`Stream` for a given device.

Args:
    device (torch.device or int, optional): selected device. Returns
        the default :class:`Stream` for the current device, given by
...

#### torch.mtia.set_stream
- **Category**: MTIA_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Set the current stream.This is a wrapper API to set the stream.
    Usage of this function is discouraged in favor of the ``stream``
    context manager.

Args:
    stream (Stream): selected stream. T...

#### torch.mtia.stream
- **Category**: MTIA_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Wrap around the Context-manager StreamContext that selects a given stream.

Arguments:
    stream (Stream): selected stream. This manager is a no-op if it's
        ``None``.
.. note:: In eager mode s...

#### torch.xpu.Stream
- **Category**: XPU_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Wrapper around a XPU stream.

A XPU stream is a linear sequence of execution that belongs to a specific
device, independent from other streams. It supports with statement as a
context manager to ensur...

#### torch.xpu.StreamContext
- **Category**: XPU_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Context-manager that selects a given stream.

All XPU kernels queued within its context will be enqueued on a selected
stream.

Args:
    Stream (Stream): selected stream. This manager is a no-op if i...

#### torch.xpu.current_stream
- **Category**: XPU_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Return the currently selected :class:`Stream` for a given device.

Args:
    device (torch.device or int, optional): selected device. Returns
        the currently selected :class:`Stream` for the cur...

#### torch.xpu.get_stream_from_external
- **Category**: XPU_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Return a :class:`Stream` from an external SYCL queue.

This function is used to wrap SYCL queue created in other libraries in order
to facilitate data exchange and multi-library interactions.

.. note...

#### torch.xpu.set_stream
- **Category**: XPU_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Set the current stream.This is a wrapper API to set the stream.
    Usage of this function is discouraged in favor of the ``stream``
    context manager.

Args:
    stream (Stream): selected stream. T...

#### torch.xpu.stream
- **Category**: XPU_OPERATIONS
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: Wrap around the Context-manager StreamContext that selects a given stream.

Arguments:
    stream (Stream): selected stream. This manager is a no-op if it's ``None``....

### 6. Neural Network Functions (Phase 3 - LOWER PRIORITY)

#### torch.channel_shuffle
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: channel_shuffle(input, groups) -> Tensor

Divide the channels in a tensor of shape :math:`(*, C , H, W)`
into g groups and rearrange them as :math:`(*, C \frac g, g, H, W)`,
while keeping the original...

#### torch.cudnn_affine_grid_generator
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_batch_norm
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_grid_sampler
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_is_acceptable
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.fake_quantize_per_channel_affine
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: fake_quantize_per_channel_affine(input, scale, zero_point, axis, quant_min, quant_max) -> Tensor

Returns a new tensor with the data in :attr:`input` fake quantized per channel using :attr:`scale`,
:a...

#### torch.hann_window
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: hann_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Hann window function.

.. math::
    w[n] = \frac{1}{2}\ \left[1 - \cos \lef...

#### torch.inner
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: inner(input, other, *, out=None) -> Tensor

Computes the dot product for 1D tensors. For higher dimensions, sums the product
of elements from :attr:`input` and :attr:`other` along their last dimension...

#### torch.miopen_rnn
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.mkldnn_rnn_layer
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.native_channel_shuffle
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: native_channel_shuffle(input, groups) -> Tensor

Native kernel level implementation of the `channel_shuffle`.
This function might become private in future releases, use with caution.

Divide the chann...

#### torch.q_per_channel_axis
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.q_per_channel_scales
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.q_per_channel_zero_points
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.quantize_per_channel
- **Category**: TORCH_CORE
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: quantize_per_channel(input, scales, zero_points, axis, dtype) -> Tensor

Converts a float tensor to a per-channel quantized tensor with given scales and zero points.

Arguments:
    input (Tensor): fl...

#### torch.quantized_rnn_tanh_cell
- **Category**: TORCH_MATH_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.rnn_tanh
- **Category**: TORCH_MATH_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.rnn_tanh_cell
- **Category**: TORCH_MATH_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_convolution
- **Category**: TORCH_NN_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_convolution_add_relu
- **Category**: TORCH_NN_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_convolution_relu
- **Category**: TORCH_NN_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.cudnn_convolution_transpose
- **Category**: TORCH_NN_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.mkldnn_adaptive_avg_pool2d
- **Category**: TORCH_NN_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.mkldnn_convolution
- **Category**: TORCH_NN_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.mkldnn_linear_backward_weights
- **Category**: TORCH_NN_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.mkldnn_max_pool2d
- **Category**: TORCH_NN_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.mkldnn_max_pool3d
- **Category**: TORCH_NN_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.quantized_rnn_relu_cell
- **Category**: TORCH_NN_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.rnn_relu
- **Category**: TORCH_NN_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.rnn_relu_cell
- **Category**: TORCH_NN_OPS
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch._numpy.hanning
- **Category**: CORE_TORCH
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch._numpy.inner
- **Category**: CORE_TORCH
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch._prims_common.are_strides_like_channels_last
- **Category**: CORE_TORCH
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch._prims_common.is_channels_last_contiguous
- **Category**: CORE_TORCH
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: True when a tensor is channels-last contiguous.

This requires that:

  - the tensor is conceptually either 4 (NHWC) or 5 (NDHWC) dimensions
  - if we name the tensor's dimensions NCHW or NCDHW, then ...

#### torch._prims_common.is_channels_last_contiguous_2d
- **Category**: CORE_TORCH
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch._prims_common.is_channels_last_contiguous_3d
- **Category**: CORE_TORCH
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch._prims_common.make_channels_last_1d_strides_for
- **Category**: CORE_TORCH
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch._prims_common.make_channels_last_2d_strides_for
- **Category**: CORE_TORCH
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch._prims_common.make_channels_last_3d_strides_for
- **Category**: CORE_TORCH
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch._prims_common.make_channels_last_strides_for
- **Category**: CORE_TORCH
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.jit.ONNXTracedModule
- **Category**: JIT_COMPILATION
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the sub...

#### torch.jit.annotate
- **Category**: JIT_COMPILATION
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Use to give type of `the_value` in TorchScript compiler.

This method is a pass-through function that returns `the_value`, used to hint TorchScript
compiler the type of `the_value`. It is a no-op when...

#### torch.jit.enable_onednn_fusion
- **Category**: JIT_COMPILATION
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Enable or disables onednn JIT fusion based on the parameter `enabled`....

#### torch.jit.onednn_fusion_enabled
- **Category**: JIT_COMPILATION
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Return whether onednn JIT fusion is enabled....

#### torch.nn.AdaptiveAvgPool1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 1D adaptive average pooling over an input signal composed of several input planes.

The output size is :math:`L_{out}`, for any input size.
The number of output features is equal to the numb...

#### torch.nn.AdaptiveAvgPool2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 2D adaptive average pooling over an input signal composed of several input planes.

The output is of size H x W, for any input size.
The number of output features is equal to the number of i...

#### torch.nn.AdaptiveAvgPool3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 3D adaptive average pooling over an input signal composed of several input planes.

The output is of size D x H x W, for any input size.
The number of output features is equal to the number ...

#### torch.nn.AdaptiveLogSoftmaxWithLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Efficient softmax approximation.

As described in
`Efficient softmax approximation for GPUs by Edouard Grave, Armand Joulin,
Moustapha Cissé, David Grangier, and Hervé Jégou
<https://arxiv.org/abs/160...

#### torch.nn.AdaptiveMaxPool1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 1D adaptive max pooling over an input signal composed of several input planes.

The output size is :math:`L_{out}`, for any input size.
The number of output features is equal to the number o...

#### torch.nn.AdaptiveMaxPool2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 2D adaptive max pooling over an input signal composed of several input planes.

The output is of size :math:`H_{out} \times W_{out}`, for any input size.
The number of output features is equ...

#### torch.nn.AdaptiveMaxPool3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 3D adaptive max pooling over an input signal composed of several input planes.

The output is of size :math:`D_{out} \times H_{out} \times W_{out}`, for any input size.
The number of output ...

#### torch.nn.AlphaDropout
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies Alpha Dropout over the input.

Alpha Dropout is a type of Dropout that maintains the self-normalizing
property.
For an input with zero mean and unit standard deviation, the output of
Alpha Dro...

#### torch.nn.AvgPool1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 1D average pooling over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size :math:`(N, C, L)`,
output :math:`(N, C, L_{out}...

#### torch.nn.AvgPool2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 2D average pooling over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
output :math:`(N, C, H_{o...

#### torch.nn.AvgPool3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 3D average pooling over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
output :math:`(N, C, D...

#### torch.nn.BCELoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Creates a criterion that measures the Binary Cross Entropy between the target and
the input probabilities:

The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

.. ...

#### torch.nn.BCEWithLogitsLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: This loss combines a `Sigmoid` layer and the `BCELoss` in one single
class. This version is more numerically stable than using a plain `Sigmoid`
followed by a `BCELoss` as, by combining the operations...

#### torch.nn.BatchNorm1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies Batch Normalization over a 2D or 3D input.

Method described in the paper
`Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift <https://arxiv.org/abs/1...

#### torch.nn.BatchNorm2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies Batch Normalization over a 4D input.

4D is a mini-batch of 2D inputs
with additional channel dimension. Method described in the paper
`Batch Normalization: Accelerating Deep Network Training ...

#### torch.nn.BatchNorm3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies Batch Normalization over a 5D input.

5D is a mini-batch of 3D inputs with additional channel dimension as described in the paper
`Batch Normalization: Accelerating Deep Network Training by Re...

#### torch.nn.Bilinear
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a bilinear transformation to the incoming data: :math:`y = x_1^T A x_2 + b`.

Args:
    in1_features: size of each first input sample, must be > 0
    in2_features: size of each second input s...

#### torch.nn.Buffer
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A kind of Tensor that should not be considered a model
parameter. For example, BatchNorm's ``running_mean`` is not a parameter, but is part of the module's state.

Buffers are :class:`~torch.Tensor` s...

#### torch.nn.CELU
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the CELU function element-wise.

.. math::
    \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

More details can be found in the paper `Continuously Differentiable Exponent...

#### torch.nn.CTCLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: The Connectionist Temporal Classification loss.

Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the
probability of possible alignments of input...

#### torch.nn.ChannelShuffle
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Divides and rearranges the channels in a tensor.

This operation divides the channels in a tensor of shape :math:`(N, C, *)`
into g groups as :math:`(N, \frac{C}{g}, g, *)` and shuffles them,
while re...

#### torch.nn.CircularPad1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor using circular padding of the input boundary.

Tensor values at the beginning of the dimension are used to pad the end,
and values at the end are used to pad the beginning. If ne...

#### torch.nn.CircularPad2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor using circular padding of the input boundary.

Tensor values at the beginning of the dimension are used to pad the end,
and values at the end are used to pad the beginning. If ne...

#### torch.nn.CircularPad3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor using circular padding of the input boundary.

Tensor values at the beginning of the dimension are used to pad the end,
and values at the end are used to pad the beginning. If ne...

#### torch.nn.ConstantPad1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor boundaries with a constant value.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args:
    padding (int, tuple): the size of the padding. If is `int`, uses...

#### torch.nn.ConstantPad2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor boundaries with a constant value.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args:
    padding (int, tuple): the size of the padding. If is `int`, uses...

#### torch.nn.ConstantPad3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor boundaries with a constant value.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args:
    padding (int, tuple): the size of the padding. If is `int`, uses...

#### torch.nn.Container
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the sub...

#### torch.nn.Conv1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 1D convolution over an input signal composed of several input
planes.

In the simplest case, the output value of the layer with input size
:math:`(N, C_{\text{in}}, L)` and output :math:`(N,...

#### torch.nn.Conv2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 2D convolution over an input signal composed of several input
planes.

In the simplest case, the output value of the layer with input size
:math:`(N, C_{\text{in}}, H, W)` and output :math:`...

#### torch.nn.Conv3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 3D convolution over an input signal composed of several input
planes.

In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
and output :math:`(N, ...

#### torch.nn.ConvTranspose1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 1D transposed convolution operator over an input image
composed of several input planes.

This module can be seen as the gradient of Conv1d with respect to its input.
It is also known as a f...

#### torch.nn.ConvTranspose2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 2D transposed convolution operator over an input image
composed of several input planes.

This module can be seen as the gradient of Conv2d with respect to its input.
It is also known as a f...

#### torch.nn.ConvTranspose3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 3D transposed convolution operator over an input image composed of several input
planes.
The transposed convolution operator multiplies each input value element-wise by a learnable kernel,
a...

#### torch.nn.CosineEmbeddingLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Creates a criterion that measures the loss given input tensors
:math:`x_1`, :math:`x_2` and a `Tensor` label :math:`y` with values 1 or -1.
Use (:math:`y=1`) to maximize the cosine similarity of two i...

#### torch.nn.CosineSimilarity
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Returns cosine similarity between :math:`x_1` and :math:`x_2`, computed along `dim`.

.. math ::
    \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilo...

#### torch.nn.CrossEntropyLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: This criterion computes the cross entropy loss between input logits
and target.

It is useful when training a classification problem with `C` classes.
If provided, the optional argument :attr:`weight`...

#### torch.nn.CrossMapLRN2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the sub...

#### torch.nn.DataParallel
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Implements data parallelism at the module level.

This container parallelizes the application of the given :attr:`module` by
splitting the input across the specified devices by chunking in the batch
d...

#### torch.nn.Dropout
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p`.

The zeroed elements are chosen independently for each forward call and are sampled from a Bernoul...

#### torch.nn.Dropout1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Randomly zero out entire channels.

A channel is a 1D feature map,
e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
batched input is a 1D tensor :math:`\text{input}[i, j]`.

Each chann...

#### torch.nn.Dropout2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Randomly zero out entire channels.

A channel is a 2D feature map,
e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
batched input is a 2D tensor :math:`\text{input}[i, j]`.

Each chann...

#### torch.nn.Dropout3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Randomly zero out entire channels.

A channel is a 3D feature map,
e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
batched input is a 3D tensor :math:`\text{input}[i, j]`.

Each chann...

#### torch.nn.ELU
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the Exponential Linear Unit (ELU) function, element-wise.

Method described in the paper: `Fast and Accurate Deep Network Learning by Exponential Linear
Units (ELUs) <https://arxiv.org/abs/151...

#### torch.nn.Embedding
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A simple lookup table that stores embeddings of a fixed dictionary and size.

This module is often used to store word embeddings and retrieve them using indices.
The input to the module is a list of i...

#### torch.nn.EmbeddingBag
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute sums or means of 'bags' of embeddings, without instantiating the intermediate embeddings.

For bags of constant length, no :attr:`per_sample_weights`, no indices equal to :attr:`padding_idx`,
...

#### torch.nn.FeatureAlphaDropout
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Randomly masks out entire channels.

A channel is a feature map,
e.g. the :math:`j`-th channel of the :math:`i`-th sample in the batch input
is a tensor :math:`\text{input}[i, j]` of the input tensor)...

#### torch.nn.Flatten
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Flattens a contiguous range of dims into a tensor.

For use with :class:`~nn.Sequential`, see :meth:`torch.flatten` for details.

Shape:
    - Input: :math:`(*, S_{\text{start}},..., S_{i}, ..., S_{\t...

#### torch.nn.Fold
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Combines an array of sliding local blocks into a large containing tensor.

Consider a batched :attr:`input` tensor containing sliding local blocks,
e.g., patches of images, of shape :math:`(N, C \time...

#### torch.nn.FractionalMaxPool2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 2D fractional max pooling over an input signal composed of several input planes.

Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

The max-po...

#### torch.nn.FractionalMaxPool3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 3D fractional max pooling over an input signal composed of several input planes.

Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

The max-po...

#### torch.nn.GELU
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the Gaussian Error Linear Units function.

.. math:: \text{GELU}(x) = x * \Phi(x)

where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

When the approximat...

#### torch.nn.GLU
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the gated linear unit function.

:math:`{GLU}(a, b)= a \otimes \sigma(b)` where :math:`a` is the first half
of the input matrices and :math:`b` is the second half.

Args:
    dim (int): the di...

#### torch.nn.GRU
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: __init__(input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0.0,bidirectional=False,device=None,dtype=None)

Apply a multi-layer gated recurrent unit (GRU) RNN to an input sequenc...

#### torch.nn.GRUCell
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A gated recurrent unit (GRU) cell.

.. math::

    \begin{array}{ll}
    r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
    z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
    n = \tanh(W_{...

#### torch.nn.GaussianNLLLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Gaussian negative log likelihood loss.

The targets are treated as samples from Gaussian distributions with
expectations and variances predicted by the neural network. For a
``target`` tensor modelled...

#### torch.nn.GroupNorm
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies Group Normalization over a mini-batch of inputs.

This layer implements the operation as described in
the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

.. math::
    y = \f...

#### torch.nn.Hardshrink
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the Hard Shrinkage (Hardshrink) function element-wise.

Hardshrink is defined as:

.. math::
    \text{HardShrink}(x) =
    \begin{cases}
    x, & \text{ if } x > \lambda \\
    x, & \text{ if...

#### torch.nn.Hardsigmoid
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the Hardsigmoid function element-wise.

Hardsigmoid is defined as:

.. math::
    \text{Hardsigmoid}(x) = \begin{cases}
        0 & \text{if~} x \le -3, \\
        1 & \text{if~} x \ge +3, \\
...

#### torch.nn.Hardswish
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the Hardswish function, element-wise.

Method described in the paper: `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_.

Hardswish is defined as:

.. math::
    \text{Hardswish}...

#### torch.nn.Hardtanh
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the HardTanh function element-wise.

HardTanh is defined as:

.. math::
    \text{HardTanh}(x) = \begin{cases}
        \text{max\_val} & \text{ if } x > \text{ max\_val } \\
        \text{min\...

#### torch.nn.HingeEmbeddingLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Measures the loss given an input tensor :math:`x` and a labels tensor :math:`y`
(containing 1 or -1).
This is usually used for measuring whether two inputs are similar or
dissimilar, e.g. using the L1...

#### torch.nn.HuberLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Creates a criterion that uses a squared term if the absolute
element-wise error falls below delta and a delta-scaled L1 term otherwise.
This loss combines advantages of both :class:`L1Loss` and :class...

#### torch.nn.Identity
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A placeholder identity operator that is argument-insensitive.

Args:
    args: any argument (unused)
    kwargs: any keyword argument (unused)

Shape:
    - Input: :math:`(*)`, where :math:`*` means a...

#### torch.nn.InstanceNorm1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies Instance Normalization.

This operation applies Instance Normalization
over a 2D (unbatched) or 3D (batched) input as described in the paper
`Instance Normalization: The Missing Ingredient for...

#### torch.nn.InstanceNorm2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies Instance Normalization.

This operation applies Instance Normalization
over a 4D input (a mini-batch of 2D inputs
with additional channel dimension) as described in the paper
`Instance Normali...

#### torch.nn.InstanceNorm3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies Instance Normalization.

This operation applies Instance Normalization
over a 5D input (a mini-batch of 3D inputs with additional channel dimension) as described in the paper
`Instance Normali...

#### torch.nn.KLDivLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: The Kullback-Leibler divergence loss.

For tensors of the same shape :math:`y_{\text{pred}},\ y_{\text{true}}`,
where :math:`y_{\text{pred}}` is the :attr:`input` and :math:`y_{\text{true}}` is the
:a...

#### torch.nn.L1Loss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Creates a criterion that measures the mean absolute error (MAE) between each element in
the input :math:`x` and target :math:`y`.

The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss ca...

#### torch.nn.LPPool1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 1D power-average pooling over an input signal composed of several input planes.

On each window, the function computed is:

.. math::
    f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}

- At p = :mat...

#### torch.nn.LPPool2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 2D power-average pooling over an input signal composed of several input planes.

On each window, the function computed is:

.. math::
    f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}

- At p = :mat...

#### torch.nn.LPPool3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 3D power-average pooling over an input signal composed of several input planes.

On each window, the function computed is:

.. math::
    f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}

- At p = :mat...

#### torch.nn.LSTM
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: __init__(input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0.0,bidirectional=False,proj_size=0,device=None,dtype=None)

Apply a multi-layer long short-term memory (LSTM) RNN to a...

#### torch.nn.LSTMCell
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A long short-term memory (LSTM) cell.

.. math::

    \begin{array}{ll}
    i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
    f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
    g = \tanh(...

#### torch.nn.LayerNorm
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies Layer Normalization over a mini-batch of inputs.

This layer implements the operation as described in
the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

.. math::
    y = \f...

#### torch.nn.LazyBatchNorm1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A :class:`torch.nn.BatchNorm1d` module with lazy initialization.

Lazy initialization based on the ``num_features`` argument of the :class:`BatchNorm1d` that is inferred
from the ``input.size(1)``.
Th...

#### torch.nn.LazyBatchNorm2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A :class:`torch.nn.BatchNorm2d` module with lazy initialization.

Lazy initialization is done for the ``num_features`` argument of the :class:`BatchNorm2d` that is inferred
from the ``input.size(1)``....

#### torch.nn.LazyBatchNorm3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A :class:`torch.nn.BatchNorm3d` module with lazy initialization.

Lazy initialization is done for the ``num_features`` argument of the :class:`BatchNorm3d` that is inferred
from the ``input.size(1)``....

#### torch.nn.LazyConv1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A :class:`torch.nn.Conv1d` module with lazy initialization of the ``in_channels`` argument.

The ``in_channels`` argument of the :class:`Conv1d` is inferred from the ``input.size(1)``.
The attributes ...

#### torch.nn.LazyConv2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A :class:`torch.nn.Conv2d` module with lazy initialization of the ``in_channels`` argument.

The ``in_channels`` argument of the :class:`Conv2d` that is inferred from the ``input.size(1)``.
The attrib...

#### torch.nn.LazyConv3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A :class:`torch.nn.Conv3d` module with lazy initialization of the ``in_channels`` argument.

The ``in_channels`` argument of the :class:`Conv3d` that is inferred from
the ``input.size(1)``.
The attrib...

#### torch.nn.LazyConvTranspose1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A :class:`torch.nn.ConvTranspose1d` module with lazy initialization of the ``in_channels`` argument.

The ``in_channels`` argument of the :class:`ConvTranspose1d` that is inferred from
the ``input.siz...

#### torch.nn.LazyConvTranspose2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A :class:`torch.nn.ConvTranspose2d` module with lazy initialization of the ``in_channels`` argument.

The ``in_channels`` argument of the :class:`ConvTranspose2d` is inferred from
the ``input.size(1)`...

#### torch.nn.LazyConvTranspose3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A :class:`torch.nn.ConvTranspose3d` module with lazy initialization of the ``in_channels`` argument.

The ``in_channels`` argument of the :class:`ConvTranspose3d` is inferred from
the ``input.size(1)`...

#### torch.nn.LazyInstanceNorm1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A :class:`torch.nn.InstanceNorm1d` module with lazy initialization of the ``num_features`` argument.

The ``num_features`` argument of the :class:`InstanceNorm1d` is inferred from the ``input.size(1)`...

#### torch.nn.LazyInstanceNorm2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A :class:`torch.nn.InstanceNorm2d` module with lazy initialization of the ``num_features`` argument.

The ``num_features`` argument of the :class:`InstanceNorm2d` is inferred from the ``input.size(1)`...

#### torch.nn.LazyInstanceNorm3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A :class:`torch.nn.InstanceNorm3d` module with lazy initialization of the ``num_features`` argument.

The ``num_features`` argument of the :class:`InstanceNorm3d` is inferred from the ``input.size(1)`...

#### torch.nn.LazyLinear
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A :class:`torch.nn.Linear` module where `in_features` is inferred.

In this module, the `weight` and `bias` are of :class:`torch.nn.UninitializedParameter`
class. They will be initialized after the fi...

#### torch.nn.LeakyReLU
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the LeakyReLU function element-wise.

.. math::
    \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)


or

.. math::
    \text{LeakyReLU}(x) =
    \begin{cases}
    x, & ...

#### torch.nn.Linear
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

On certain ROCm devices, when using float16 inputs this...

#### torch.nn.LocalResponseNorm
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies local response normalization over an input signal.

The input signal is composed of several input planes, where channels occupy the second dimension.
Applies normalization across channels.

.....

#### torch.nn.LogSigmoid
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the Logsigmoid function element-wise.

.. math::
    \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)

Shape:
    - Input: :math:`(*)`, where :math:`*` means any number of dim...

#### torch.nn.LogSoftmax
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional input Tensor.

The LogSoftmax formulation can be simplified as:

.. math::
    \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(...

#### torch.nn.MSELoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Creates a criterion that measures the mean squared error (squared L2 norm) between
each element in the input :math:`x` and target :math:`y`.

The unreduced (i.e. with :attr:`reduction` set to ``'none'...

#### torch.nn.MarginRankingLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Creates a criterion that measures the loss given
inputs :math:`x1`, :math:`x2`, two 1D mini-batch or 0D `Tensors`,
and a label 1D mini-batch or 0D `Tensor` :math:`y` (containing 1 or -1).

If :math:`y...

#### torch.nn.MaxPool1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 1D max pooling over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size :math:`(N, C, L)`
and output :math:`(N, C, L_{out})...

#### torch.nn.MaxPool2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 2D max pooling over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
output :math:`(N, C, H_{out},...

#### torch.nn.MaxPool3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 3D max pooling over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
output :math:`(N, C, D_{ou...

#### torch.nn.MaxUnpool1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Computes a partial inverse of :class:`MaxPool1d`.

:class:`MaxPool1d` is not fully invertible, since the non-maximal values are lost.

:class:`MaxUnpool1d` takes in as input the output of :class:`MaxP...

#### torch.nn.MaxUnpool2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Computes a partial inverse of :class:`MaxPool2d`.

:class:`MaxPool2d` is not fully invertible, since the non-maximal values are lost.

:class:`MaxUnpool2d` takes in as input the output of :class:`MaxP...

#### torch.nn.MaxUnpool3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Computes a partial inverse of :class:`MaxPool3d`.

:class:`MaxPool3d` is not fully invertible, since the non-maximal values are lost.
:class:`MaxUnpool3d` takes in as input the output of :class:`MaxPo...

#### torch.nn.Mish
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the Mish function, element-wise.

Mish: A Self Regularized Non-Monotonic Neural Activation Function.

.. math::
    \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

.. note::
    See `Mis...

#### torch.nn.Module
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the sub...

#### torch.nn.ModuleDict
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Holds submodules in a dictionary.

:class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
but modules it contains are properly registered, and will be visible by all
:class:`~t...

#### torch.nn.ModuleList
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Holds submodules in a list.

:class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
modules it contains are properly registered, and will be visible by all
:class:`~torch.nn.Modu...

#### torch.nn.MultiLabelMarginLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Creates a criterion that optimizes a multi-class multi-classification
hinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`)
and output :math:`y` (which is a 2D `Tensor` of ...

#### torch.nn.MultiLabelSoftMarginLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Creates a criterion that optimizes a multi-label one-versus-all
loss based on max-entropy, between input :math:`x` and target :math:`y` of size
:math:`(N, C)`.
For each sample in the minibatch:

.. ma...

#### torch.nn.MultiMarginLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Creates a criterion that optimizes a multi-class classification hinge
loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`) and
output :math:`y` (which is a 1D tensor of target c...

#### torch.nn.MultiheadAttention
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Allows the model to jointly attend to information from different representation subspaces.

.. note::
    See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.htm...

#### torch.nn.NLLLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: The negative log likelihood loss. It is useful to train a classification
problem with `C` classes.

If provided, the optional argument :attr:`weight` should be a 1D Tensor assigning
weight to each of ...

#### torch.nn.NLLLoss2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: The negative log likelihood loss. It is useful to train a classification
problem with `C` classes.

If provided, the optional argument :attr:`weight` should be a 1D Tensor assigning
weight to each of ...

#### torch.nn.PReLU
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the element-wise PReLU function.

.. math::
    \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

or

.. math::
    \text{PReLU}(x) =
    \begin{cases}
    x, & \text{ if } x \ge 0 \\
    ax, & \te...

#### torch.nn.PairwiseDistance
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Computes the pairwise distance between input vectors, or between columns of input matrices.

Distances are computed using ``p``-norm, with constant ``eps`` added to avoid division by zero
if ``p`` is ...

#### torch.nn.Parameter
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A kind of Tensor that is to be considered a module parameter.

Parameters are :class:`~torch.Tensor` subclasses, that have a
very special property when used with :class:`Module` s - when they're
assig...

#### torch.nn.ParameterDict
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Holds parameters in a dictionary.

ParameterDict can be indexed like a regular Python dictionary, but Parameters it
contains are properly registered, and will be visible by all Module methods.
Other o...

#### torch.nn.ParameterList
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Holds parameters in a list.

:class:`~torch.nn.ParameterList` can be used like a regular Python
list, but Tensors that are :class:`~torch.nn.Parameter` are properly registered,
and will be visible by ...

#### torch.nn.PixelShuffle
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Rearrange elements in a tensor according to an upscaling factor.

Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
to a tensor of shape :math:`(*, C, H \times r, W \times r)`, ...

#### torch.nn.PixelUnshuffle
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Reverse the PixelShuffle operation.

Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements
in a tensor of shape :math:`(*, C, H \times r, W \times r)` to a tensor of shape
:ma...

#### torch.nn.PoissonNLLLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Negative log likelihood loss with Poisson distribution of target.

The loss can be described as:

.. math::
    \text{target} \sim \mathrm{Poisson}(\text{input})

    \text{loss}(\text{input}, \text{t...

#### torch.nn.RMSNorm
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

This layer implements the operation as described in
the paper `Root Mean Square Layer Normalization <https://arxiv.org/pdf/19...

#### torch.nn.RNN
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: __init__(input_size,hidden_size,num_layers=1,nonlinearity='tanh',bias=True,batch_first=False,dropout=0.0,bidirectional=False,device=None,dtype=None)

Apply a multi-layer Elman RNN with :math:`\tanh` o...

#### torch.nn.RNNBase
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Base class for RNN modules (RNN, LSTM, GRU).

Implements aspects of RNNs shared by the RNN, LSTM, and GRU classes, such as module initialization
and utility methods for parameter storage management.

...

#### torch.nn.RNNCell
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: An Elman RNN cell with tanh or ReLU non-linearity.

.. math::

    h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})

If :attr:`nonlinearity` is `'relu'`, then ReLU is used in place of tanh.

Args:
...

#### torch.nn.RNNCellBase
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the sub...

#### torch.nn.RReLU
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the randomized leaky rectified linear unit function, element-wise.

Method described in the paper:
`Empirical Evaluation of Rectified Activations in Convolutional Network <https://arxiv.org/ab...

#### torch.nn.ReLU
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the rectified linear unit function element-wise.

:math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

Args:
    inplace: can optionally do the operation in-place. Default: ``False``

Shape:
    - Inp...

#### torch.nn.ReLU6
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the ReLU6 function element-wise.

.. math::
    \text{ReLU6}(x) = \min(\max(0,x), 6)

Args:
    inplace: can optionally do the operation in-place. Default: ``False``

Shape:
    - Input: :math...

#### torch.nn.ReflectionPad1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor using the reflection of the input boundary.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args:
    padding (int, tuple): the size of the padding. If is `...

#### torch.nn.ReflectionPad2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor using the reflection of the input boundary.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args:
    padding (int, tuple): the size of the padding. If is `...

#### torch.nn.ReflectionPad3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor using the reflection of the input boundary.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args:
    padding (int, tuple): the size of the padding. If is `...

#### torch.nn.ReplicationPad1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor using replication of the input boundary.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args:
    padding (int, tuple): the size of the padding. If is `int...

#### torch.nn.ReplicationPad2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor using replication of the input boundary.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args:
    padding (int, tuple): the size of the padding. If is `int...

#### torch.nn.ReplicationPad3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor using replication of the input boundary.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args:
    padding (int, tuple): the size of the padding. If is `int...

#### torch.nn.SELU
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the SELU function element-wise.

.. math::
    \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))

with :math:`\alpha = 1.6732632423543772848170429916717` and
:math:...

#### torch.nn.Sequential
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A sequential container.

Modules will be added to it in the order they are passed in the
constructor. Alternatively, an ``OrderedDict`` of modules can be
passed in. The ``forward()`` method of ``Seque...

#### torch.nn.SiLU
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the Sigmoid Linear Unit (SiLU) function, element-wise.

The SiLU function is also known as the swish function.

.. math::
    \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is ...

#### torch.nn.Sigmoid
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the Sigmoid function element-wise.

.. math::
    \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}


Shape:
    - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
    ...

#### torch.nn.SmoothL1Loss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Creates a criterion that uses a squared term if the absolute
element-wise error falls below beta and an L1 term otherwise.
It is less sensitive to outliers than :class:`torch.nn.MSELoss` and in some c...

#### torch.nn.SoftMarginLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Creates a criterion that optimizes a two-class classification
logistic loss between input tensor :math:`x` and target tensor :math:`y`
(containing 1 or -1).

.. math::
    \text{loss}(x, y) = \sum_i \...

#### torch.nn.Softmax
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the Softmax function to an n-dimensional input Tensor.

Rescales them so that the elements of the n-dimensional output Tensor
lie in the range [0,1] and sum to 1.

Softmax is defined as:

.. m...

#### torch.nn.Softmax2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies SoftMax over features to each spatial location.

When given an image of ``Channels x Height x Width``, it will
apply `Softmax` to each location :math:`(Channels, h_i, w_j)`

Shape:
    - Input...

#### torch.nn.Softmin
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the Softmin function to an n-dimensional input Tensor.

Rescales them so that the elements of the n-dimensional output Tensor
lie in the range `[0, 1]` and sum to 1.

Softmin is defined as:

....

#### torch.nn.Softplus
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the Softplus function element-wise.

.. math::
    \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))

SoftPlus is a smooth approximation to the ReLU function and can be used
to ...

#### torch.nn.Softshrink
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the soft shrinkage function element-wise.

.. math::
    \text{SoftShrinkage}(x) =
    \begin{cases}
    x - \lambda, & \text{ if } x > \lambda \\
    x + \lambda, & \text{ if } x < -\lambda \...

#### torch.nn.Softsign
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the element-wise Softsign function.

.. math::
    \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

Shape:
    - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
    - Output: :math...

#### torch.nn.SyncBatchNorm
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies Batch Normalization over a N-Dimensional input.

The N-D input is a mini-batch of [N-2]D inputs with additional channel dimension) as described in the paper
`Batch Normalization: Accelerating ...

#### torch.nn.Tanh
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the Hyperbolic Tangent (Tanh) function element-wise.

Tanh is defined as:

.. math::
    \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

Shape:
    - Input: :math:`...

#### torch.nn.Tanhshrink
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies the element-wise Tanhshrink function.

.. math::
    \text{Tanhshrink}(x) = x - \tanh(x)

Shape:
    - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
    - Output: :math:`...

#### torch.nn.Threshold
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Thresholds each element of the input Tensor.

Threshold is defined as:

.. math::
    y =
    \begin{cases}
    x, &\text{ if } x > \text{threshold} \\
    \text{value}, &\text{ otherwise }
    \end{c...

#### torch.nn.Transformer
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A transformer model.

.. note::
    See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    for an in depth discussion of the performant building blocks ...

#### torch.nn.TransformerDecoder
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: TransformerDecoder is a stack of N decoder layers.

.. note::
    See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    for an in depth discussion of t...

#### torch.nn.TransformerDecoderLayer
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

.. note::
    See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html...

#### torch.nn.TransformerEncoder
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: TransformerEncoder is a stack of N encoder layers.

.. note::
    See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    for an in depth discussion of t...

#### torch.nn.TransformerEncoderLayer
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: TransformerEncoderLayer is made up of self-attn and feedforward network.

.. note::
    See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    for an in...

#### torch.nn.TripletMarginLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Creates a criterion that measures the triplet loss given an input
tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
This is used for measuring a relative sim...

#### torch.nn.TripletMarginWithDistanceLoss
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Creates a criterion that measures the triplet loss given input
tensors :math:`a`, :math:`p`, and :math:`n` (representing anchor,
positive, and negative examples, respectively), and a nonnegative,
real...

#### torch.nn.Unflatten
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Unflattens a tensor dim expanding it to a desired shape. For use with :class:`~nn.Sequential`.

* :attr:`dim` specifies the dimension of the input tensor to be unflattened, and it can
  be either `int...

#### torch.nn.Unfold
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Extracts sliding local blocks from a batched input tensor.

Consider a batched :attr:`input` tensor of shape :math:`(N, C, *)`,
where :math:`N` is the batch dimension, :math:`C` is the channel dimensi...

#### torch.nn.UninitializedBuffer
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A buffer that is not initialized.

Uninitialized Buffer is a a special case of :class:`torch.Tensor`
where the shape of the data is still unknown.

Unlike a :class:`torch.Tensor`, uninitialized parame...

#### torch.nn.UninitializedParameter
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A parameter that is not initialized.

Uninitialized Parameters are a special case of :class:`torch.nn.Parameter`
where the shape of the data is still unknown.

Unlike a :class:`torch.nn.Parameter`, un...

#### torch.nn.Upsample
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

The input data is assumed to be of the form
`minibatch x channels x [optional depth] x [optional height] x width`....

#### torch.nn.UpsamplingBilinear2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 2D bilinear upsampling to an input signal composed of several input channels.

To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
as it's constructor argument...

#### torch.nn.UpsamplingNearest2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 2D nearest neighbor upsampling to an input signal composed of several input channels.

To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
as it's constructor ...

#### torch.nn.ZeroPad1d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor boundaries with zero.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args:
    padding (int, tuple): the size of the padding. If is `int`, uses the same
  ...

#### torch.nn.ZeroPad2d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor boundaries with zero.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args:
    padding (int, tuple): the size of the padding. If is `int`, uses the same
  ...

#### torch.nn.ZeroPad3d
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Pads the input tensor boundaries with zero.

For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

Args:
    padding (int, tuple): the size of the padding. If is `int`, uses the same
  ...

#### torch.nn.factory_kwargs
- **Category**: NEURAL_NETWORK
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Return a canonicalized dict of factory kwargs.

Given kwargs, returns a canonicalized dict of factory kwargs that can be directly passed
to factory functions like torch.empty, or errors if unrecognize...

#### torch.nn.functional.DType
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, thi...

#### torch.nn.functional.adaptive_avg_pool1d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: adaptive_avg_pool1d(input, output_size) -> Tensor

Applies a 1D adaptive average pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveAvgPool1d` for details an...

#### torch.nn.functional.adaptive_avg_pool2d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a 2D adaptive average pooling over an input signal composed of several input planes.

See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

Args:
    output_size: the target ou...

#### torch.nn.functional.adaptive_avg_pool3d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a 3D adaptive average pooling over an input signal composed of several input planes.

See :class:`~torch.nn.AdaptiveAvgPool3d` for details and output shape.

Args:
    output_size: the target ou...

#### torch.nn.functional.adaptive_max_pool1d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: adaptive_max_pool1d(input, output_size, return_indices=False)

Applies a 1D adaptive max pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveMaxPool1d` for de...

#### torch.nn.functional.adaptive_max_pool1d_with_indices
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: adaptive_max_pool1d(input, output_size, return_indices=False)

Applies a 1D adaptive max pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveMaxPool1d` for de...

#### torch.nn.functional.adaptive_max_pool2d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: adaptive_max_pool2d(input, output_size, return_indices=False)

Applies a 2D adaptive max pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveMaxPool2d` for de...

#### torch.nn.functional.adaptive_max_pool2d_with_indices
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: adaptive_max_pool2d(input, output_size, return_indices=False)

Applies a 2D adaptive max pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveMaxPool2d` for de...

#### torch.nn.functional.adaptive_max_pool3d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: adaptive_max_pool3d(input, output_size, return_indices=False)

Applies a 3D adaptive max pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveMaxPool3d` for de...

#### torch.nn.functional.adaptive_max_pool3d_with_indices
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: adaptive_max_pool3d(input, output_size, return_indices=False)

Applies a 3D adaptive max pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveMaxPool3d` for de...

#### torch.nn.functional.affine_grid
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Generate 2D or 3D flow field (sampling grid), given a batch of affine matrices :attr:`theta`.

.. note::
    This function is often used in conjunction with :func:`grid_sample`
    to build `Spatial T...

#### torch.nn.functional.alpha_dropout
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply alpha dropout to the input.

See :class:`~torch.nn.AlphaDropout` for details....

#### torch.nn.functional.assert_int_or_pair
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: ...

#### torch.nn.functional.avg_pool1d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor

Applies a 1D average pooling over an input signal composed of several
input planes.

See :cla...

#### torch.nn.functional.avg_pool2d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor

Applies 2D average-pooling operation in :math:`kH \times kW` regions b...

#### torch.nn.functional.avg_pool3d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor

Applies 3D average-pooling operation in :math:`kT \times kH \times kW`...

#### torch.nn.functional.batch_norm
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply Batch Normalization for each channel across a batch of data.

See :class:`~torch.nn.BatchNorm1d`, :class:`~torch.nn.BatchNorm2d`,
:class:`~torch.nn.BatchNorm3d` for details....

#### torch.nn.functional.bilinear
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: bilinear(input1, input2, weight, bias=None) -> Tensor

Applies a bilinear transformation to the incoming data:
:math:`y = x_1^T A x_2 + b`

Shape:

    - input1: :math:`(N, *, H_{in1})` where :math:`H...

#### torch.nn.functional.binary_cross_entropy
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute Binary Cross Entropy between the target and input probabilities.

See :class:`~torch.nn.BCELoss` for details.

Args:
    input: Tensor of arbitrary shape as probabilities.
    target: Tensor o...

#### torch.nn.functional.binary_cross_entropy_with_logits
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute Binary Cross Entropy between target and input logits.

See :class:`~torch.nn.BCEWithLogitsLoss` for details.

Args:
    input: Tensor of arbitrary shape as unnormalized scores (often referred ...

#### torch.nn.functional.boolean_dispatch
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Dispatches to either of 2 script functions based on a boolean argument.
In TorchScript, the boolean argument must be constant so that the correct
function to use can be determined at compile time....

#### torch.nn.functional.celu
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: celu(input, alpha=1., inplace=False) -> Tensor

Applies element-wise,
:math:`\text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))`.

See :class:`~torch.nn.CELU` for more details....

#### torch.nn.functional.celu_
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: celu_(input, alpha=1.) -> Tensor

In-place version of :func:`~celu`....

#### torch.nn.functional.channel_shuffle
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: channel_shuffle(input, groups) -> Tensor

Divide the channels in a tensor of shape :math:`(*, C , H, W)`
into g groups and rearrange them as :math:`(*, C \frac g, g, H, W)`,
while keeping the original...

#### torch.nn.functional.conv1d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

This operator supports :ref:`Ten...

#### torch.nn.functional.conv2d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 2D convolution over an input image composed of several input
planes.

This operator supports :ref:`Tens...

#### torch.nn.functional.conv3d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 3D convolution over an input image composed of several input
planes.

This operator supports :ref:`Tens...

#### torch.nn.functional.conv_tbc
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Applies a 1-dimensional sequence convolution over an input sequence.
Input and output dimensions are (Time, Batch, Channels) - hence TBC.

Args:
    input: input tensor of shape :math:`(\text{sequence...

#### torch.nn.functional.conv_transpose1d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 1D transposed convolution operator over an input signal
composed of several...

#### torch.nn.functional.conv_transpose2d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 2D transposed convolution operator over an input image
composed of several ...

#### torch.nn.functional.conv_transpose3d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 3D transposed convolution operator over an input image
composed of several ...

#### torch.nn.functional.cosine_embedding_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the cosine embedding loss.

See :class:`~torch.nn.CosineEmbeddingLoss` for details.

Args:
   input1 (Tensor): Predicted values.
   input2 (Tensor): Predicted values.
   target (Tensor): Groun...

#### torch.nn.functional.cosine_similarity
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: cosine_similarity(x1, x2, dim=1, eps=1e-8) -> Tensor

Returns cosine similarity between ``x1`` and ``x2``, computed along dim. ``x1`` and ``x2`` must be broadcastable
to a common shape. ``dim`` refers...

#### torch.nn.functional.cross_entropy
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the cross entropy loss between input logits and target.

See :class:`~torch.nn.CrossEntropyLoss` for details.

Args:
    input (Tensor) : Predicted unnormalized logits;
        see Shape secti...

#### torch.nn.functional.ctc_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the Connectionist Temporal Classification loss.

See :class:`~torch.nn.CTCLoss` for details.

Note:
    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator...

#### torch.nn.functional.dropout
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: During training, randomly zeroes some elements of the input tensor with probability :attr:`p`.

Uses samples from a Bernoulli distribution.

See :class:`~torch.nn.Dropout` for details.

Args:
    p: p...

#### torch.nn.functional.dropout1d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Randomly zero out entire channels (a channel is a 1D feature map).

For example, the :math:`j`-th channel of the :math:`i`-th sample in the
batched input is a 1D tensor :math:`\text{input}[i, j]` of t...

#### torch.nn.functional.dropout2d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Randomly zero out entire channels (a channel is a 2D feature map).

For example, the :math:`j`-th channel of the :math:`i`-th sample in the
batched input is a 2D tensor :math:`\text{input}[i, j]` of t...

#### torch.nn.functional.dropout3d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Randomly zero out entire channels (a channel is a 3D feature map).

For example, the :math:`j`-th channel of the :math:`i`-th sample in the
batched input is a 3D tensor :math:`\text{input}[i, j]` of t...

#### torch.nn.functional.elu
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply the Exponential Linear Unit (ELU) function element-wise.

See :class:`~torch.nn.ELU` for more details....

#### torch.nn.functional.elu_
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: elu_(input, alpha=1.) -> Tensor

In-place version of :func:`~elu`....

#### torch.nn.functional.embedding
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Generate a simple lookup table that looks up embeddings in a fixed dictionary and size.

This module is often used to retrieve word embeddings using indices.
The input to the module is a list of indic...

#### torch.nn.functional.embedding_bag
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute sums, means or maxes of `bags` of embeddings.

Calculation is done without instantiating the intermediate embeddings.
See :class:`torch.nn.EmbeddingBag` for more details.

Note:
    This opera...

#### torch.nn.functional.feature_alpha_dropout
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Randomly masks out entire channels (a channel is a feature map).

For example, the :math:`j`-th channel of the :math:`i`-th sample in the batch input
is a tensor :math:`\text{input}[i, j]` of the inpu...

#### torch.nn.functional.fold
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Combine an array of sliding local blocks into a large containing tensor.

.. warning::
    Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.

See :class:`torch.nn...

#### torch.nn.functional.fractional_max_pool2d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

Applies 2D fractional max pooling over an input signal composed of several i...

#### torch.nn.functional.fractional_max_pool2d_with_indices
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

Applies 2D fractional max pooling over an input signal composed of several i...

#### torch.nn.functional.fractional_max_pool3d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

Applies 3D fractional max pooling over an input signal composed of several i...

#### torch.nn.functional.fractional_max_pool3d_with_indices
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

Applies 3D fractional max pooling over an input signal composed of several i...

#### torch.nn.functional.gaussian_nll_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the Gaussian negative log likelihood loss.

See :class:`~torch.nn.GaussianNLLLoss` for details.

Args:
    input: Expectation of the Gaussian distribution.
    target: Sample from the Gaussian...

#### torch.nn.functional.gelu
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: gelu(input, approximate = 'none') -> Tensor

When the approximate argument is 'none', it applies element-wise the function
:math:`\text{GELU}(x) = x * \Phi(x)`

where :math:`\Phi(x)` is the Cumulative...

#### torch.nn.functional.glu
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: glu(input, dim=-1) -> Tensor

The gated linear unit. Computes:

.. math ::
    \text{GLU}(a, b) = a \otimes \sigma(b)

where `input` is split in half along `dim` to form `a` and `b`, :math:`\sigma`
is...

#### torch.nn.functional.grid_sample
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute grid sample.

Given an :attr:`input` and a flow-field :attr:`grid`, computes the
``output`` using :attr:`input` values and pixel locations from :attr:`grid`.

Currently, only spatial (4-D) and...

#### torch.nn.functional.group_norm
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply Group Normalization for last certain number of dimensions.

See :class:`~torch.nn.GroupNorm` for details....

#### torch.nn.functional.gumbel_softmax
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Sample from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretize.

Args:
  logits: `[..., num_features]` unnormalized log probabilities
  tau: non-negative scalar temperatu...

#### torch.nn.functional.handle_torch_function
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Implement a function with checks for ``__torch_function__`` overrides.

See torch::autograd::handle_torch_function for the equivalent of this
function in the C++ implementation.

Arguments
---------
p...

#### torch.nn.functional.hardshrink
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: hardshrink(input, lambd=0.5) -> Tensor

Applies the hard shrinkage function element-wise

See :class:`~torch.nn.Hardshrink` for more details....

#### torch.nn.functional.hardsigmoid
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply the Hardsigmoid function element-wise.

.. math::
    \text{Hardsigmoid}(x) = \begin{cases}
        0 & \text{if~} x \le -3, \\
        1 & \text{if~} x \ge +3, \\
        x / 6 + 1 / 2 & \text{...

#### torch.nn.functional.hardswish
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply hardswish function, element-wise.

Follows implementation as described in the paper:
`Searching for MobileNetV3`_.

.. math::
    \text{Hardswish}(x) = \begin{cases}
        0 & \text{if~} x \le...

#### torch.nn.functional.hardtanh
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Tensor

Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more
details....

#### torch.nn.functional.hardtanh_
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: hardtanh_(input, min_val=-1., max_val=1.) -> Tensor

In-place version of :func:`~hardtanh`....

#### torch.nn.functional.has_torch_function
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Check for __torch_function__ implementations in the elements of an iterable
or if a __torch_function__ mode is enabled.  Considers exact ``Tensor`` s
and ``Parameter`` s non-dispatchable.  Use this to...

#### torch.nn.functional.has_torch_function_unary
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Special case of `has_torch_function` for single inputs.
Instead of:
  `has_torch_function((t,))`
call:
  `has_torch_function_unary(t)`
which skips unnecessary packing and unpacking work....

#### torch.nn.functional.has_torch_function_variadic
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Special case of `has_torch_function` that skips tuple creation.

This uses the METH_FASTCALL protocol introduced in Python 3.7

Instead of:
  `has_torch_function((a, b))`
call:
  `has_torch_function_v...

#### torch.nn.functional.hinge_embedding_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the hinge embedding loss.

See :class:`~torch.nn.HingeEmbeddingLoss` for details.

Args:
   input (Tensor): Predicted values.
   target (Tensor): Ground truth values.
   margin (float, optiona...

#### torch.nn.functional.huber_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the Huber loss, with optional weighting.

Function uses a squared term if the absolute
element-wise error falls below delta and a delta-scaled L1 term otherwise.

When delta equals 1, this los...

#### torch.nn.functional.instance_norm
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply Instance Normalization independently for each channel in every data sample within a batch.

See :class:`~torch.nn.InstanceNorm1d`, :class:`~torch.nn.InstanceNorm2d`,
:class:`~torch.nn.InstanceNo...

#### torch.nn.functional.interpolate
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Down/up samples the input.

Tensor interpolated to either the given :attr:`size` or the given
:attr:`scale_factor`

The algorithm used for interpolation is determined by :attr:`mode`.

Currently tempo...

#### torch.nn.functional.kl_div
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the KL Divergence loss.

Refer - The `Kullback-Leibler divergence Loss
<https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`__

See :class:`~torch.nn.KLDivLoss` for details.

Args:
    ...

#### torch.nn.functional.l1_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the L1 loss, with optional weighting.

Function that takes the mean element-wise absolute value difference.

See :class:`~torch.nn.L1Loss` for details.

Args:
    input (Tensor): Predicted val...

#### torch.nn.functional.layer_norm
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply Layer Normalization for last certain number of dimensions.

See :class:`~torch.nn.LayerNorm` for details....

#### torch.nn.functional.leaky_relu
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor

Applies element-wise,
:math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`

See :class:`~torch.nn.LeakyReLU` ...

#### torch.nn.functional.leaky_relu_
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: leaky_relu_(input, negative_slope=0.01) -> Tensor

In-place version of :func:`~leaky_relu`....

#### torch.nn.functional.linear
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: linear(input, weight, bias=None) -> Tensor

Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

This operation supports 2-D :attr:`weight` with :ref:`sparse layout<sparse-docs...

#### torch.nn.functional.local_response_norm
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply local response normalization over an input signal.

The input signal is composed of several input planes, where channels occupy the second dimension.
Normalization is applied across channels.

S...

#### torch.nn.functional.log_softmax
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a softmax followed by a logarithm.

While mathematically equivalent to log(softmax(x)), doing these two
operations separately is slower and numerically unstable. This function
uses an alternativ...

#### torch.nn.functional.logsigmoid
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: logsigmoid(input) -> Tensor

Applies element-wise :math:`\text{LogSigmoid}(x_i) = \log \left(\frac{1}{1 + \exp(-x_i)}\right)`

See :class:`~torch.nn.LogSigmoid` for more details....

#### torch.nn.functional.lp_pool1d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a 1D power-average pooling over an input signal composed of several input planes.

If the sum of all inputs to the power of `p` is
zero, the gradient is set to zero as well.

See :class:`~torch....

#### torch.nn.functional.lp_pool2d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a 2D power-average pooling over an input signal composed of several input planes.

If the sum of all inputs to the power of `p` is
zero, the gradient is set to zero as well.

See :class:`~torch....

#### torch.nn.functional.lp_pool3d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a 3D power-average pooling over an input signal composed of several input planes.

If the sum of all inputs to the power of `p` is
zero, the gradient is set to zero as well.

See :class:`~torch....

#### torch.nn.functional.margin_ranking_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the margin ranking loss.

See :class:`~torch.nn.MarginRankingLoss` for details.

Args:
    input1 (Tensor): Predicted values.
    input2 (Tensor): Predicted values.
    target (Tensor): Ground...

#### torch.nn.functional.max_pool1d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

Applies a 1D max pooling over an input signal composed of several input
planes.

.. note::
  ...

#### torch.nn.functional.max_pool1d_with_indices
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

Applies a 1D max pooling over an input signal composed of several input
planes.

.. note::
  ...

#### torch.nn.functional.max_pool2d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

Applies a 2D max pooling over an input signal composed of several input
planes.

.. note::
  ...

#### torch.nn.functional.max_pool2d_with_indices
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

Applies a 2D max pooling over an input signal composed of several input
planes.

.. note::
  ...

#### torch.nn.functional.max_pool3d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

Applies a 3D max pooling over an input signal composed of several input
planes.

.. note::
  ...

#### torch.nn.functional.max_pool3d_with_indices
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

Applies a 3D max pooling over an input signal composed of several input
planes.

.. note::
  ...

#### torch.nn.functional.max_unpool1d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute a partial inverse of :class:`MaxPool1d`.

See :class:`~torch.nn.MaxUnpool1d` for details....

#### torch.nn.functional.max_unpool2d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute a partial inverse of :class:`MaxPool2d`.

See :class:`~torch.nn.MaxUnpool2d` for details....

#### torch.nn.functional.max_unpool3d
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute a partial inverse of :class:`MaxPool3d`.

See :class:`~torch.nn.MaxUnpool3d` for details....

#### torch.nn.functional.mish
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply the Mish function, element-wise.

Mish: A Self Regularized Non-Monotonic Neural Activation Function.

.. math::
    \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

.. note::
    See `Mish:...

#### torch.nn.functional.mse_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the element-wise mean squared error, with optional weighting.

See :class:`~torch.nn.MSELoss` for details.

Args:
    input (Tensor): Predicted values.
    target (Tensor): Ground truth values...

#### torch.nn.functional.multi_head_attention_forward
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Forward method for MultiHeadAttention.

.. note::
    See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    for an in depth discussion of the performan...

#### torch.nn.functional.multi_margin_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the multi margin loss, with optional weighting.

See :class:`~torch.nn.MultiMarginLoss` for details.

Args:
    input (Tensor): Predicted values.
    target (Tensor): Ground truth values.
    ...

#### torch.nn.functional.multilabel_margin_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the multilabel margin loss.

See :class:`~torch.nn.MultiLabelMarginLoss` for details.

Args:
   input (Tensor): Predicted values.
   target (Tensor): Ground truth values.
   size_average (bool...

#### torch.nn.functional.multilabel_soft_margin_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the multilabel soft margin loss.

See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.

Args:
   input (Tensor): Predicted values.
   target (Tensor): Ground truth values.
   size_aver...

#### torch.nn.functional.native_channel_shuffle
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: native_channel_shuffle(input, groups) -> Tensor

Native kernel level implementation of the `channel_shuffle`.
This function might become private in future releases, use with caution.

Divide the chann...

#### torch.nn.functional.nll_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the negative log likelihood loss.

See :class:`~torch.nn.NLLLoss` for details.

Args:
    input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
        in case of 2D Loss...

#### torch.nn.functional.normalize
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Perform :math:`L_p` normalization of inputs over specified dimension.

For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
:math:`n_{dim}` -element vector :math:`v` along d...

#### torch.nn.functional.one_hot
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: one_hot(tensor, num_classes=-1) -> LongTensor

Takes LongTensor with index values of shape ``(*)`` and returns a tensor
of shape ``(*, num_classes)`` that have zeros everywhere except where the
index ...

#### torch.nn.functional.pad
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: pad(input, pad, mode="constant", value=None) -> Tensor

Pads tensor.

Padding size:
    The padding size by which to pad some dimensions of :attr:`input`
    are described starting from the last dimen...

#### torch.nn.functional.pairwise_distance
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False) -> Tensor

See :class:`torch.nn.PairwiseDistance` for details...

#### torch.nn.functional.pdist
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: pdist(input, p=2) -> Tensor

Computes the p-norm distance between every pair of row vectors in the input.
This is identical to the upper triangular portion, excluding the diagonal, of
`torch.norm(inpu...

#### torch.nn.functional.pixel_shuffle
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: pixel_shuffle(input, upscale_factor) -> Tensor

Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)` to a
tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is the :at...

#### torch.nn.functional.pixel_unshuffle
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: pixel_unshuffle(input, downscale_factor) -> Tensor

Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements in a
tensor of shape :math:`(*, C, H \times r, W \times r)` to a tens...

#### torch.nn.functional.poisson_nll_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the Poisson negative log likelihood loss.

See :class:`~torch.nn.PoissonNLLLoss` for details.

Args:
    input: Expectation of underlying Poisson distribution.
    target: Random sample :math:...

#### torch.nn.functional.prelu
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: prelu(input, weight) -> Tensor

Applies element-wise the function
:math:`\text{PReLU}(x) = \max(0,x) + \text{weight} * \min(0,x)` where weight is a
learnable parameter.

.. note::
    `weight` is expe...

#### torch.nn.functional.relu
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: relu(input, inplace=False) -> Tensor

Applies the rectified linear unit function element-wise. See
:class:`~torch.nn.ReLU` for more details....

#### torch.nn.functional.relu6
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: relu6(input, inplace=False) -> Tensor

Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`.

See :class:`~torch.nn.ReLU6` for more details....

#### torch.nn.functional.relu_
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: relu_(input) -> Tensor

In-place version of :func:`~relu`....

#### torch.nn.functional.rms_norm
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply Root Mean Square Layer Normalization.

See :class:`~torch.nn.RMSNorm` for details....

#### torch.nn.functional.rrelu
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Tensor

Randomized leaky ReLU.

See :class:`~torch.nn.RReLU` for more details....

#### torch.nn.functional.rrelu_
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: rrelu_(input, lower=1./8, upper=1./3, training=False) -> Tensor

In-place version of :func:`~rrelu`....

#### torch.nn.functional.scaled_dot_product_attention
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
    is_causal=False, scale=None, enable_gqa=False) -> Tensor:

Computes scaled dot product attention on query, key and va...

#### torch.nn.functional.selu
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: selu(input, inplace=False) -> Tensor

Applies element-wise,
:math:`\text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))`,
with :math:`\alpha=1.6732632423543772848170429916717` and
:m...

#### torch.nn.functional.selu_
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: selu_(input) -> Tensor

In-place version of :func:`~selu`....

#### torch.nn.functional.sigmoid
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: sigmoid(input) -> Tensor

Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

See :class:`~torch.nn.Sigmoid` for more details....

#### torch.nn.functional.silu
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply the Sigmoid Linear Unit (SiLU) function, element-wise.

The SiLU function is also known as the swish function.

.. math::
    \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is th...

#### torch.nn.functional.smooth_l1_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the Smooth L1 loss.

Function uses a squared term if the absolute
element-wise error falls below beta and an L1 term otherwise.

See :class:`~torch.nn.SmoothL1Loss` for details.

Args:
    inp...

#### torch.nn.functional.soft_margin_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the soft margin loss.

See :class:`~torch.nn.SoftMarginLoss` for details.

Args:
   input (Tensor): Predicted values.
   target (Tensor): Ground truth values.
   size_average (bool, optional):...

#### torch.nn.functional.softmax
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a softmax function.

Softmax is defined as:

:math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

It is applied to all slices along dim, and will re-scale them so that the element...

#### torch.nn.functional.softmin
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a softmin function.

Note that :math:`\text{Softmin}(x) = \text{Softmax}(-x)`. See softmax definition for mathematical formula.

See :class:`~torch.nn.Softmin` for more details.

Args:
    input...

#### torch.nn.functional.softplus
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: softplus(input, beta=1, threshold=20) -> Tensor

Applies element-wise, the function :math:`\text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))`.

For numerical stability the implementatio...

#### torch.nn.functional.softshrink
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: softshrink(input, lambd=0.5) -> Tensor

Applies the soft shrinkage function elementwise

See :class:`~torch.nn.Softshrink` for more details....

#### torch.nn.functional.softsign
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: softsign(input) -> Tensor

Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}`

See :class:`~torch.nn.Softsign` for more details....

#### torch.nn.functional.tanh
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: tanh(input) -> Tensor

Applies element-wise,
:math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`

See :class:`~torch.nn.Tanh` for more details....

#### torch.nn.functional.tanhshrink
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: tanhshrink(input) -> Tensor

Applies element-wise, :math:`\text{Tanhshrink}(x) = x - \text{Tanh}(x)`

See :class:`~torch.nn.Tanhshrink` for more details....

#### torch.nn.functional.threshold
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Apply a threshold to each element of the input Tensor.

See :class:`~torch.nn.Threshold` for more details....

#### torch.nn.functional.threshold_
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: threshold_(input, threshold, value) -> Tensor

In-place version of :func:`~threshold`....

#### torch.nn.functional.triplet_margin_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the triplet loss between given input tensors and a margin greater than 0.

See :class:`~torch.nn.TripletMarginLoss` for details....

#### torch.nn.functional.triplet_margin_with_distance_loss
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Compute the triplet margin loss for input tensors using a custom distance function.

See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details....

#### torch.nn.functional.unfold
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Extract sliding local blocks from a batched input tensor.

.. warning::
    Currently, only 4-D input tensors (batched image-like tensors) are
    supported.

.. warning::

    More than one element o...

#### torch.nn.functional.upsample
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Upsample input.

Provided tensor is upsampled to either the given :attr:`size` or the given
:attr:`scale_factor`

.. warning::
    This function is deprecated in favor of :func:`torch.nn.functional.in...

#### torch.nn.functional.upsample_bilinear
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Upsamples the input, using bilinear upsampling.

.. warning::
    This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
    This is equivalent with
    ``nn.functional.inter...

#### torch.nn.functional.upsample_nearest
- **Category**: NEURAL_NETWORK_FUNCTIONAL
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Upsamples the input, using nearest neighbours' pixel values.

.. warning::
    This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
    This is equivalent with ``nn.functio...

#### torch.onnx.Any
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements...

#### torch.onnx.ExportOptions
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Options for dynamo_export.

.. deprecated:: 2.7
    Please use ``torch.onnx.export(..., dynamo=True)`` instead.

Attributes:
    dynamic_shapes: Shape information hint for input/output tensors.
      ...

#### torch.onnx.JitScalarType
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Scalar types defined in torch.

Use ``JitScalarType`` to convert from torch and JIT scalar types to ONNX scalar types.

Examples:
    >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
    >>> # xdocte...

#### torch.onnx.ONNXProgram
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A class to represent an ONNX program that is callable with torch tensors.

Attributes:
    model: The ONNX model as an ONNX IR model object.
    exported_program: The exported program that produced th...

#### torch.onnx.OnnxExporterError
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Errors raised by the ONNX exporter. This is the base class for all exporter errors....

#### torch.onnx.OperatorExportTypes
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Members:

ONNX

ONNX_ATEN

ONNX_ATEN_FALLBACK

ONNX_FALLTHROUGH...

#### torch.onnx.TrainingMode
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Members:

EVAL

PRESERVE

TRAINING...

#### torch.onnx.deprecated
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Indicate that a class, function or overload is deprecated.

When this decorator is applied to an object, the type checker
will generate a diagnostic on usage of the deprecated object.

Usage:

    @de...

#### torch.onnx.dynamo_export
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Export a torch.nn.Module to an ONNX graph.

.. deprecated:: 2.7
    Please use ``torch.onnx.export(..., dynamo=True)`` instead.

Args:
    model: The PyTorch model to be exported to ONNX.
    model_ar...

#### torch.onnx.enable_fake_mode
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Enable fake mode for the duration of the context.

Internally it instantiates a :class:`torch._subclasses.fake_tensor.FakeTensorMode` context manager
that converts user input and model parameters into...

#### torch.onnx.export
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Exports a model into ONNX format.

Setting ``dynamo=True`` enables the new ONNX export logic
which is based on :class:`torch.export.ExportedProgram` and a more modern
set of translation logic. This is...

#### torch.onnx.is_in_onnx_export
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Returns whether it is in the middle of ONNX export....

#### torch.onnx.is_onnxrt_backend_supported
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Returns ``True`` if ONNX Runtime dependencies are installed and usable
to support TorchDynamo backend integration; ``False`` otherwise.

Example::

    # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
  ...

#### torch.onnx.register_custom_op_symbolic
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Registers a symbolic function for a custom operator.

When the user registers symbolic for custom/contrib ops,
it is highly recommended to add shape inference for that operator via setType API,
otherw...

#### torch.onnx.select_model_mode_for_export
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: A context manager to temporarily set the training mode of ``model``
to ``mode``, resetting it when we exit the with-block.

.. deprecated:: 2.7
    Please set training mode before exporting the model....

#### torch.onnx.unregister_custom_op_symbolic
- **Category**: ONNX_EXPORT
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Unregisters ``symbolic_name``.

See "Custom Operators" in the module documentation for an example usage.

Args:
    symbolic_name (str): The name of the custom operator in "<domain>::<op>"
        for...

#### torch.quantization.MovingAveragePerChannelMinMaxObserver
- **Category**: QUANTIZATION
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Observer module for computing the quantization parameters based on the
running per channel min and max values.

This observer uses the tensor min/max statistics to compute the per channel
quantization...

#### torch.quantization.PerChannelMinMaxObserver
- **Category**: QUANTIZATION
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: Observer module for computing the quantization parameters based on the
running per channel min and max values.

This observer uses the tensor min/max statistics to compute the per channel
quantization...

### 7. Device-Specific Functions (Phase 3 - LOWER PRIORITY)

#### torch.DispatchKey
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Members:

Undefined

CompositeExplicitAutogradNonFunctional

CompositeExplicitAutograd

CompositeImplicitAutogradNestedTensor

CompositeImplicitAutograd

AutogradNestedTensor

AutogradOther

Autograd
...

#### torch.Generator
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Generator(device='cpu') -> Generator

Creates and returns a generator object that manages the state of the algorithm which
produces pseudo random numbers. Used as a keyword argument in many :ref:`inpl...

#### torch.GradScaler
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: An instance ``scaler`` of :class:`GradScaler`.

Helps perform the steps of gradient scaling
conveniently.

* ``scaler.scale(loss)`` multiplies a given loss by ``scaler``'s current scale factor.
* ``sc...

#### torch.Tag
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Members:

core

cudagraph_unsafe

data_dependent_output

dynamic_output_shape

flexible_layout

generated

inplace_view

maybe_aliasing_or_mutating

needs_exact_strides

needs_fixed_stride_order

nond...

#### torch.add
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: add(input, other, *, alpha=1, out=None) -> Tensor

Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.

.. math::
    \text{{out}}_i = \text{{input}}_i + \text{{alpha}} \times \text{{other}...

#### torch.addbmm
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: addbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored
in :attr:`batch1` and :attr:`batch2`,
with a reduced add step (all matr...

#### torch.addcdiv
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: addcdiv(input, tensor1, tensor2, *, value=1, out=None) -> Tensor

Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`,
multiplies the result by the scalar :attr:`value` and adds i...

#### torch.addcmul
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: addcmul(input, tensor1, tensor2, *, value=1, out=None) -> Tensor

Performs the element-wise multiplication of :attr:`tensor1`
by :attr:`tensor2`, multiplies the result by the scalar :attr:`value`
and ...

#### torch.addmm
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: addmm(input, mat1, mat2, out_dtype=None, *, beta=1, alpha=1, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.
The matrix :attr:`input` is added to t...

#### torch.addmv
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: addmv(input, mat, vec, *, beta=1, alpha=1, out=None) -> Tensor

Performs a matrix-vector product of the matrix :attr:`mat` and
the vector :attr:`vec`.
The vector :attr:`input` is added to the final re...

#### torch.addr
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: addr(input, vec1, vec2, *, beta=1, alpha=1, out=None) -> Tensor

Performs the outer-product of vectors :attr:`vec1` and :attr:`vec2`
and adds it to the matrix :attr:`input`.

Optional values :attr:`be...

#### torch.adjoint
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: adjoint(input: Tensor) -> Tensor
Returns a view of the tensor conjugated and with the last two dimensions transposed.

``x.adjoint()`` is equivalent to ``x.transpose(-2, -1).conj()`` for complex tenso...

#### torch.alias_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.alias`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.all
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: all(input: Tensor) -> Tensor

Tests if all elements in :attr:`input` evaluate to `True`.

.. note:: This function matches the behaviour of NumPy in returning
          output of dtype `bool` for all s...

#### torch.allclose
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: allclose(input: Tensor, other: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool

This function checks if :attr:`input` and :attr:`other` satisfy the condition:

.. ma...

#### torch.amax
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: amax(input, dim, keepdim=False, *, out=None) -> Tensor

Returns the maximum value of each slice of the :attr:`input` tensor in the given
dimension(s) :attr:`dim`.

.. note::
    The difference between...

#### torch.amin
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: amin(input, dim, keepdim=False, *, out=None) -> Tensor

Returns the minimum value of each slice of the :attr:`input` tensor in the given
dimension(s) :attr:`dim`.

.. note::
    The difference between...

#### torch.aminmax
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: aminmax(input, *, dim=None, keepdim=False, out=None) -> (Tensor min, Tensor max)

Computes the minimum and maximum values of the :attr:`input` tensor.

Args:
    input (Tensor):
        The input tens...

#### torch.angle
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: angle(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Computes the element-wise angle (in radians) of the given :attr:`input` tensor.

.. math::
    \text{out}_{i} = angle(\text{input}_{i})

Args:...

#### torch.any
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: any(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Tests if any element in :attr:`input` evaluates to `True`.

.. note:: This function matches the behaviour of NumPy in returning
          output...

#### torch.argmax
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: argmax(input) -> LongTensor

Returns the indices of the maximum value of all elements in the :attr:`input` tensor.

This is the second value returned by :meth:`torch.max`. See its
documentation for th...

#### torch.argmin
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: argmin(input, dim=None, keepdim=False) -> LongTensor

Returns the indices of the minimum value(s) of the flattened tensor or along a dimension

This is the second value returned by :meth:`torch.min`. ...

#### torch.argsort
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: argsort(input, dim=-1, descending=False, stable=False) -> Tensor

Returns the indices that sort a tensor along a given dimension in ascending
order by value.

This is the second value returned by :met...

#### torch.argwhere
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: argwhere(input) -> Tensor

Returns a tensor containing the indices of all non-zero elements of
:attr:`input`.  Each row in the result contains the indices of a non-zero
element in :attr:`input`. The r...

#### torch.as_strided
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: as_strided(input, size, stride, storage_offset=None) -> Tensor

Create a view of an existing `torch.Tensor` :attr:`input` with specified
:attr:`size`, :attr:`stride` and :attr:`storage_offset`.

.. wa...

#### torch.as_strided_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.as_strided`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.as_strided_scatter
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: as_strided_scatter(input, src, size, stride, storage_offset=None) -> Tensor

Embeds the values of the :attr:`src` tensor into :attr:`input` along
the elements corresponding to the result of calling
in...

#### torch.asarray
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: asarray(obj: Any, *, dtype: Optional[dtype], device: Optional[DeviceLikeType], copy: Optional[bool] = None, requires_grad: bool = False) -> Tensor # noqa: B950

Converts :attr:`obj` to a tensor.

:att...

#### torch.atleast_1d
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns a 1-dimensional view of each input tensor with zero dimensions.
Input tensors with one or more dimensions are returned as-is.

Args:
    input (Tensor or list of Tensors)

Returns:
    output ...

#### torch.atleast_2d
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns a 2-dimensional view of each input tensor with zero dimensions.
Input tensors with two or more dimensions are returned as-is.

Args:
    input (Tensor or list of Tensors)

Returns:
    output ...

#### torch.atleast_3d
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns a 3-dimensional view of each input tensor with zero dimensions.
Input tensors with three or more dimensions are returned as-is.

Args:
    input (Tensor or list of Tensors)

Returns:
    outpu...

#### torch.autocast
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Instances of :class:`autocast` serve as context managers or decorators that
allow regions of your script to run in mixed precision.

In these regions, ops run in an op-specific dtype chosen by autocas...

#### torch.baddbmm
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: baddbmm(input, batch1, batch2, out_dtype=None, *, beta=1, alpha=1, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices in :attr:`batch1`
and :attr:`batch2`.
:attr:`input` is added ...

#### torch.bartlett_window
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bartlett_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Bartlett window function.

.. math::
    w[n] = 1 - \left| \frac{2n}{N-1...

#### torch.bincount
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bincount(input, weights=None, minlength=0) -> Tensor

Count the frequency of each value in an array of non-negative ints.

The number of bins (size 1) is one larger than the largest value in
:attr:`in...

#### torch.bitwise_and
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bitwise_and(input, other, *, out=None) -> Tensor

Computes the bitwise AND of :attr:`input` and :attr:`other`. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the ...

#### torch.bitwise_left_shift
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bitwise_left_shift(input, other, *, out=None) -> Tensor

Computes the left arithmetic shift of :attr:`input` by :attr:`other` bits.
The input tensor must be of integral type. This operator supports
:r...

#### torch.bitwise_not
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bitwise_not(input, *, out=None) -> Tensor

Computes the bitwise NOT of the given input tensor. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical NOT.

Ar...

#### torch.bitwise_or
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bitwise_or(input: Tensor, other: Tensor, *, out: Optional[Tensor]) -> Tensor

Computes the bitwise OR of :attr:`input` and :attr:`other`. The input tensor must be of
integral or Boolean types. For boo...

#### torch.bitwise_right_shift
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bitwise_right_shift(input, other, *, out=None) -> Tensor

Computes the right arithmetic shift of :attr:`input` by :attr:`other` bits.
The input tensor must be of integral type. This operator supports
...

#### torch.bitwise_xor
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bitwise_xor(input, other, *, out=None) -> Tensor

Computes the bitwise XOR of :attr:`input` and :attr:`other`. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the ...

#### torch.blackman_window
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: blackman_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Blackman window function.

.. math::
    w[n] = 0.42 - 0.5 \cos \left( \...

#### torch.block_diag
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Create a block diagonal matrix from provided tensors.

Args:
    *tensors: One or more tensors with 0, 1, or 2 dimensions.

Returns:
    Tensor: A 2 dimensional tensor with all the input tensors arran...

#### torch.bmm
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bmm(input, mat2, out_dtype=None, *, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored in :attr:`input`
and :attr:`mat2`.

:attr:`input` and :attr:`mat2` must be 3-D tensor...

#### torch.broadcast_shapes
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: broadcast_shapes(*shapes) -> Size

Similar to :func:`broadcast_tensors` but for shapes.

This is equivalent to
``torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape``
but avoids the need creat...

#### torch.broadcast_to
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: broadcast_to(input, shape) -> Tensor

Broadcasts :attr:`input` to the shape :attr:`\shape`.
Equivalent to calling ``input.expand(shape)``. See :meth:`~Tensor.expand` for details.

Args:
    input (Ten...

#### torch.bucketize
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bucketize(input, boundaries, *, out_int32=False, right=False, out=None) -> Tensor

Returns the indices of the buckets to which each value in the :attr:`input` belongs, where the
boundaries of the buck...

#### torch.cartesian_prod
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Do cartesian product of the given sequence of tensors. The behavior is similar to
python's `itertools.product`.

Args:
    *tensors: any number of 1 dimensional tensors.

Returns:
    Tensor: A tensor...

#### torch.cat
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cat(tensors, dim=0, *, out=None) -> Tensor

Concatenates the given sequence of tensors in :attr:`tensors` in the given dimension.
All tensors must either have the same shape (except in the concatenati...

#### torch.cdist
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Computes batched the p-norm distance between each pair of the two collections of row vectors.

Args:
    x1 (Tensor): input tensor where the last two dimensions represent the points and the feature di...

#### torch.celu_
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: celu_(input, alpha=1.) -> Tensor

In-place version of :func:`~celu`....

#### torch.chain_matmul
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the matrix product of the :math:`N` 2-D tensors. This product is efficiently computed
using the matrix chain order algorithm which selects the order in which incurs the lowest cost in terms
of...

#### torch.cholesky
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cholesky(input, upper=False, *, out=None) -> Tensor

Computes the Cholesky decomposition of a symmetric positive-definite
matrix :math:`A` or for batches of symmetric positive-definite matrices.

If :...

#### torch.cholesky_inverse
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cholesky_inverse(L, upper=False, *, out=None) -> Tensor

Computes the inverse of a complex Hermitian or real symmetric
positive-definite matrix given its Cholesky decomposition.

Let :math:`A` be a co...

#### torch.cholesky_solve
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cholesky_solve(B, L, upper=False, *, out=None) -> Tensor

Computes the solution of a system of linear equations with complex Hermitian
or real symmetric positive-definite lhs given its Cholesky decomp...

#### torch.chunk
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: chunk(input: Tensor, chunks: int, dim: int = 0) -> Tuple[Tensor, ...]

Attempts to split a tensor into the specified number of chunks. Each chunk is a view of
the input tensor.


.. note::

    This f...

#### torch.clamp
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: clamp(input, min=None, max=None, *, out=None) -> Tensor

Clamps all elements in :attr:`input` into the range `[` :attr:`min`, :attr:`max` `]`.
Letting min_value and max_value be :attr:`min` and :attr:...

#### torch.clip
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: clip(input, min=None, max=None, *, out=None) -> Tensor

Alias for :func:`torch.clamp`....

#### torch.clone
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: clone(input, *, memory_format=torch.preserve_format) -> Tensor

Returns a copy of :attr:`input`.

.. note::

    This function is differentiable, so gradients will flow back from the
    result of thi...

#### torch.col_indices_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.col_indices`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.column_stack
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: column_stack(tensors, *, out=None) -> Tensor

Creates a new tensor by horizontally stacking the tensors in :attr:`tensors`.

Equivalent to ``torch.hstack(tensors)``, except each zero or one dimensiona...

#### torch.combinations
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: combinations(input: Tensor, r: int = 2, with_replacement: bool = False) -> seq

Compute combinations of length :math:`r` of the given tensor. The behavior is similar to
python's `itertools.combination...

#### torch.compile
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Optimizes given model/function using TorchDynamo and specified backend.
If you are compiling an :class:`torch.nn.Module`, you can also use :meth:`torch.nn.Module.compile`
to compile the module inplace...

#### torch.complex
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: complex(real, imag, *, out=None) -> Tensor

Constructs a complex tensor with its real part equal to :attr:`real` and its
imaginary part equal to :attr:`imag`.

Args:
    real (Tensor): The real part o...

#### torch.concat
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: concat(tensors, dim=0, *, out=None) -> Tensor

Alias of :func:`torch.cat`....

#### torch.concatenate
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: concatenate(tensors, axis=0, out=None) -> Tensor

Alias of :func:`torch.cat`....

#### torch.cond
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Conditionally applies `true_fn` or `false_fn`.

.. warning::
    `torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types and
    doesn't support training curr...

#### torch.conj
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: conj(input) -> Tensor

Returns a view of :attr:`input` with a flipped conjugate bit. If :attr:`input` has a non-complex dtype,
this function just returns :attr:`input`.

.. note::
    :func:`torch.con...

#### torch.conj_physical
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: conj_physical(input, *, out=None) -> Tensor

Computes the element-wise conjugate of the given :attr:`input` tensor.
If :attr:`input` has a non-complex dtype, this function just returns :attr:`input`.
...

#### torch.copysign
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: copysign(input, other, *, out=None) -> Tensor

Create a new floating-point tensor with the magnitude of :attr:`input` and the sign of :attr:`other`, elementwise.

.. math::
    \text{out}_{i} = \begin...

#### torch.corrcoef
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: corrcoef(input) -> Tensor

Estimates the Pearson product-moment correlation coefficient matrix of the variables given by the :attr:`input` matrix,
where rows are the variables and columns are the obse...

#### torch.count_nonzero
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: count_nonzero(input, dim=None) -> Tensor

Counts the number of non-zero values in the tensor :attr:`input` along the given :attr:`dim`.
If no dim is specified then all non-zeros in the tensor are coun...

#### torch.cov
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cov(input, *, correction=1, fweights=None, aweights=None) -> Tensor

Estimates the covariance matrix of the variables given by the :attr:`input` matrix, where rows are
the variables and columns are th...

#### torch.cross
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cross(input, other, dim=None, *, out=None) -> Tensor


Returns the cross product of vectors in dimension :attr:`dim` of :attr:`input`
and :attr:`other`.

Supports input of float, double, cfloat and cd...

#### torch.crow_indices_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.crow_indices`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.cummax
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cummax(input, dim, *, out=None) -> (Tensor, LongTensor)
Returns a namedtuple ``(values, indices)`` where ``values`` is the cumulative maximum of
elements of :attr:`input` in the dimension :attr:`dim`....

#### torch.cummin
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cummin(input, dim, *, out=None) -> (Tensor, LongTensor)
Returns a namedtuple ``(values, indices)`` where ``values`` is the cumulative minimum of
elements of :attr:`input` in the dimension :attr:`dim`....

#### torch.cumprod
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cumprod(input, dim, *, dtype=None, out=None) -> Tensor

Returns the cumulative product of elements of :attr:`input` in the dimension
:attr:`dim`.

For example, if :attr:`input` is a vector of size N, ...

#### torch.cumsum
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cumsum(input, dim, *, dtype=None, out=None) -> Tensor

Returns the cumulative sum of elements of :attr:`input` in the dimension
:attr:`dim`.

For example, if :attr:`input` is a vector of size N, the r...

#### torch.cumulative_trapezoid
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cumulative_trapezoid(y, x=None, *, dx=None, dim=-1) -> Tensor

Cumulatively computes the `trapezoidal rule <https://en.wikipedia.org/wiki/Trapezoidal_rule>`_
along :attr:`dim`. By default the spacing ...

#### torch.deg2rad
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: deg2rad(input, *, out=None) -> Tensor

Returns a new tensor with each of the elements of :attr:`input`
converted from angles in degrees to radians.

Args:
    input (Tensor): the input tensor.

Keywor...

#### torch.dequantize
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: dequantize(tensor) -> Tensor

Returns an fp32 Tensor by dequantizing a quantized Tensor

Args:
    tensor (Tensor): A quantized Tensor

.. function:: dequantize(tensors) -> sequence of Tensors
   :noi...

#### torch.det
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: det(input) -> Tensor

Alias for :func:`torch.linalg.det`...

#### torch.detach_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.detach`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.diag
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: diag(input, diagonal=0, *, out=None) -> Tensor

- If :attr:`input` is a vector (1-D tensor), then returns a 2-D square tensor
  with the elements of :attr:`input` as the diagonal.
- If :attr:`input` i...

#### torch.diag_embed
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: diag_embed(input, offset=0, dim1=-2, dim2=-1) -> Tensor

Creates a tensor whose diagonals of certain 2D planes (specified by
:attr:`dim1` and :attr:`dim2`) are filled by :attr:`input`.
To facilitate c...

#### torch.diagflat
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: diagflat(input, offset=0) -> Tensor

- If :attr:`input` is a vector (1-D tensor), then returns a 2-D square tensor
  with the elements of :attr:`input` as the diagonal.
- If :attr:`input` is a tensor ...

#### torch.diagonal
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: diagonal(input, offset=0, dim1=0, dim2=1) -> Tensor

Returns a partial view of :attr:`input` with the its diagonal elements
with respect to :attr:`dim1` and :attr:`dim2` appended as a dimension
at the...

#### torch.diagonal_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.diagonal`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.diagonal_scatter
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: diagonal_scatter(input, src, offset=0, dim1=0, dim2=1) -> Tensor

Embeds the values of the :attr:`src` tensor into :attr:`input` along
the diagonal elements of :attr:`input`, with respect to :attr:`di...

#### torch.diff
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: diff(input, n=1, dim=-1, prepend=None, append=None) -> Tensor

Computes the n-th forward difference along the given dimension.

The first-order differences are given by `out[i] = input[i + 1] - input[...

#### torch.digamma
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: digamma(input, *, out=None) -> Tensor

Alias for :func:`torch.special.digamma`....

#### torch.dist
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: dist(input, other, p=2) -> Tensor

Returns the p-norm of (:attr:`input` - :attr:`other`)

The shapes of :attr:`input` and :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>`.

Args:
  ...

#### torch.div
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: div(input, other, *, rounding_mode=None, out=None) -> Tensor

Divides each element of the input ``input`` by the corresponding element of
:attr:`other`.

.. math::
    \text{out}_i = \frac{\text{input...

#### torch.divide
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: divide(input, other, *, rounding_mode=None, out=None) -> Tensor

Alias for :func:`torch.div`....

#### torch.dot
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: dot(input, tensor, *, out=None) -> Tensor

Computes the dot product of two 1D tensors.

.. note::

    Unlike NumPy's dot, torch.dot intentionally only supports computing the dot product
    of two 1D...

#### torch.dsplit
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: dsplit(input, indices_or_sections) -> List of Tensors

Splits :attr:`input`, a tensor with three or more dimensions, into multiple tensors
depthwise according to :attr:`indices_or_sections`. Each spli...

#### torch.dstack
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: dstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence depthwise (along third axis).

This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been reshaped b...

#### torch.einsum
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: einsum(equation, *operands) -> Tensor

Sums the product of the elements of the input :attr:`operands` along dimensions specified using a notation
based on the Einstein summation convention.

Einsum al...

#### torch.enable_grad
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Context-manager that enables gradient calculation.

Enables gradient calculation, if it has been disabled via :class:`~no_grad`
or :class:`~set_grad_enabled`.

This context manager is thread local; it...

#### torch.eq
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: eq(input, other, *, out=None) -> Tensor

Computes element-wise equality

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first arg...

#### torch.equal
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: equal(input, other) -> bool

``True`` if two tensors have the same size and elements, ``False`` otherwise.

.. note::

    Tensors containing NaNs are never equal to each other. Additionally, this fun...

#### torch.erf
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: erf(input, *, out=None) -> Tensor

Alias for :func:`torch.special.erf`....

#### torch.erfc
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: erfc(input, *, out=None) -> Tensor

Alias for :func:`torch.special.erfc`....

#### torch.erfinv
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: erfinv(input, *, out=None) -> Tensor

Alias for :func:`torch.special.erfinv`....

#### torch.fix
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: fix(input, *, out=None) -> Tensor

Alias for :func:`torch.trunc`...

#### torch.flatten
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: flatten(input, start_dim=0, end_dim=-1) -> Tensor

Flattens :attr:`input` by reshaping it into a one-dimensional tensor. If :attr:`start_dim` or :attr:`end_dim`
are passed, only dimensions starting wi...

#### torch.flip
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: flip(input, dims) -> Tensor

Reverse the order of an n-D tensor along given axis in dims.

.. note::
    `torch.flip` makes a copy of :attr:`input`'s data. This is different from NumPy's `np.flip`,
  ...

#### torch.fliplr
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: fliplr(input) -> Tensor

Flip tensor in the left/right direction, returning a new tensor.

Flip the entries in each row in the left/right direction.
Columns are preserved, but appear in a different or...

#### torch.flipud
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: flipud(input) -> Tensor

Flip tensor in the up/down direction, returning a new tensor.

Flip the entries in each column in the up/down direction.
Rows are preserved, but appear in a different order th...

#### torch.float_power
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: float_power(input, exponent, *, out=None) -> Tensor

Raises :attr:`input` to the power of :attr:`exponent`, elementwise, in double precision.
If neither input is complex returns a ``torch.float64`` te...

#### torch.fmax
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: fmax(input, other, *, out=None) -> Tensor

Computes the element-wise maximum of :attr:`input` and :attr:`other`.

This is like :func:`torch.maximum` except it handles NaNs differently:
if exactly one ...

#### torch.fmin
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: fmin(input, other, *, out=None) -> Tensor

Computes the element-wise minimum of :attr:`input` and :attr:`other`.

This is like :func:`torch.minimum` except it handles NaNs differently:
if exactly one ...

#### torch.fmod
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: fmod(input, other, *, out=None) -> Tensor

Applies C++'s `std::fmod <https://en.cppreference.com/w/cpp/numeric/math/fmod>`_ entrywise.
The result has the same sign as the dividend :attr:`input` and it...

#### torch.frac
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: frac(input, *, out=None) -> Tensor

Computes the fractional portion of each element in :attr:`input`.

.. math::
    \text{out}_{i} = \text{input}_{i} - \left\lfloor |\text{input}_{i}| \right\rfloor *...

#### torch.from_dlpack
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: from_dlpack(ext_tensor) -> Tensor

Converts a tensor from an external library into a ``torch.Tensor``.

The returned PyTorch tensor will share the memory with the input tensor
(which may have come fro...

#### torch.from_file
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: from_file(filename, shared=None, size=0, *, dtype=None, layout=None, device=None, pin_memory=False)

Creates a CPU tensor with a storage backed by a memory-mapped file.

If ``shared`` is True, then me...

#### torch.from_numpy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: from_numpy(ndarray) -> Tensor

Creates a :class:`Tensor` from a :class:`numpy.ndarray`.

The returned tensor and :attr:`ndarray` share the same memory. Modifications to
the tensor will be reflected in...

#### torch.frombuffer
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False) -> Tensor

Creates a 1-dimensional :class:`Tensor` from an object that implements
the Python buffer protocol.

Skips the first :at...

#### torch.full
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Creates a tensor of size :attr:`size` filled with :attr:`fill_value`. The
tensor's dt...

#### torch.full_like
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: full_like(input, fill_value, \*, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor with the same size as :attr:`input...

#### torch.gather
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: gather(input, dim, index, *, sparse_grad=False, out=None) -> Tensor

Gathers values along an axis specified by `dim`.

For a 3-D tensor the output is specified by::

    out[i][j][k] = input[index[i][...

#### torch.gcd
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: gcd(input, other, *, out=None) -> Tensor

Computes the element-wise greatest common divisor (GCD) of :attr:`input` and :attr:`other`.

Both :attr:`input` and :attr:`other` must have integer types.

.....

#### torch.ge
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ge(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} \geq \text{other}` element-wise.


The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcastin...

#### torch.geqrf
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: geqrf(input, *, out=None) -> (Tensor, Tensor)

This is a low-level function for calling LAPACK's geqrf directly. This function
returns a namedtuple (a, tau) as defined in `LAPACK documentation for geq...

#### torch.ger
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ger(input, vec2, *, out=None) -> Tensor

Alias of :func:`torch.outer`.

.. warning::
    This function is deprecated and will be removed in a future PyTorch release.
    Use :func:`torch.outer` instea...

#### torch.get_autocast_xla_dtype
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.get_num_interop_threads
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: get_num_interop_threads() -> int

Returns the number of threads used for inter-op parallelism on CPU
(e.g. in JIT interpreter)...

#### torch.get_num_threads
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: get_num_threads() -> int

Returns the number of threads used for parallelizing CPU operations...

#### torch.get_rng_state
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the random number generator state as a `torch.ByteTensor`.

.. note:: The returned state is for the default generator on CPU only.

See also: :func:`torch.random.fork_rng`....

#### torch.gradient
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: gradient(input, *, spacing=1, dim=None, edge_order=1) -> List of Tensors

Estimates the gradient of a function :math:`g : \mathbb{R}^n \rightarrow \mathbb{R}` in
one or more dimensions using the `seco...

#### torch.greater
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: greater(input, other, *, out=None) -> Tensor

Alias for :func:`torch.gt`....

#### torch.greater_equal
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: greater_equal(input, other, *, out=None) -> Tensor

Alias for :func:`torch.ge`....

#### torch.gt
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: gt(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} > \text{other}` element-wise.


The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-s...

#### torch.hamming_window
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Hamming window function.

.. math::
    w[n] = \al...

#### torch.hardshrink
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: hardshrink(input, lambd=0.5) -> Tensor

Applies the hard shrinkage function element-wise

See :class:`~torch.nn.Hardshrink` for more details....

#### torch.heaviside
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: heaviside(input, values, *, out=None) -> Tensor

Computes the Heaviside step function for each element in :attr:`input`.
The Heaviside step function is defined as:

.. math::
    \text{{heaviside}}(in...

#### torch.histc
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: histc(input, bins=100, min=0, max=0, *, out=None) -> Tensor

Computes the histogram of a tensor.

The elements are sorted into equal width bins between :attr:`min` and
:attr:`max`. If :attr:`min` and ...

#### torch.histogram
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: histogram(input, bins, *, range=None, weight=None, density=False, out=None) -> (Tensor, Tensor)

Computes a histogram of the values in a tensor.

:attr:`bins` can be an integer or a 1D tensor.

If :at...

#### torch.histogramdd
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: histogramdd(input, bins, *, range=None, weight=None, density=False, out=None) -> (Tensor, Tensor[])

Computes a multi-dimensional histogram of the values in a tensor.

Interprets the elements of an in...

#### torch.hsplit
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: hsplit(input, indices_or_sections) -> List of Tensors

Splits :attr:`input`, a tensor with one or more dimensions, into multiple tensors
horizontally according to :attr:`indices_or_sections`. Each spl...

#### torch.hspmm
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: hspmm(mat1, mat2, *, out=None) -> Tensor

Performs a matrix multiplication of a :ref:`sparse COO matrix
<sparse-coo-docs>` :attr:`mat1` and a strided matrix :attr:`mat2`. The
result is a (1 + 1)-dimen...

#### torch.hstack
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: hstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence horizontally (column wise).

This is equivalent to concatenation along the first axis for 1-D tensors, and along the second axis for a...

#### torch.hypot
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: hypot(input, other, *, out=None) -> Tensor

Given the legs of a right triangle, return its hypotenuse.

.. math::
    \text{out}_{i} = \sqrt{\text{input}_{i}^{2} + \text{other}_{i}^{2}}

The shapes of...

#### torch.i0
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: i0(input, *, out=None) -> Tensor

Alias for :func:`torch.special.i0`....

#### torch.igamma
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: igamma(input, other, *, out=None) -> Tensor

Alias for :func:`torch.special.gammainc`....

#### torch.igammac
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: igammac(input, other, *, out=None) -> Tensor

Alias for :func:`torch.special.gammaincc`....

#### torch.imag
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: imag(input) -> Tensor

Returns a new tensor containing imaginary values of the :attr:`self` tensor.
The returned tensor and :attr:`self` share the same underlying storage.

.. warning::
    :func:`ima...

#### torch.index_add
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: index_add(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[Number, _complex] = 1, out: Optional[Tensor]) -> Tensor # noqa: B950

See :meth:`~Tensor.index_add_` for function desc...

#### torch.index_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: index_copy(input: Tensor, dim: int, index: Tensor, source: Tensor, *, out: Optional[Tensor]) -> Tensor

See :meth:`~Tensor.index_add_` for function description....

#### torch.index_reduce
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: index_reduce(input: Tensor, dim: int, index: Tensor, source: Tensor, reduce: str, *, include_self: bool = True, out: Optional[Tensor]) -> Tensor # noqa: B950

See :meth:`~Tensor.index_reduce_` for fun...

#### torch.index_select
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: index_select(input, dim, index, *, out=None) -> Tensor

Returns a new tensor which indexes the :attr:`input` tensor along dimension
:attr:`dim` using the entries in :attr:`index` which is a `LongTenso...

#### torch.indices_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.indices`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.inference_mode
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Context-manager that enables or disables inference mode.

InferenceMode is a context manager analogous to :class:`~no_grad`
to be used when you are certain your operations will have no interactions
wi...

#### torch.initial_seed
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the initial seed for generating random numbers as a
Python `long`.

.. note:: The returned seed is for the default generator on CPU only....

#### torch.inverse
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: inverse(input, *, out=None) -> Tensor

Alias for :func:`torch.linalg.inv`...

#### torch.is_autocast_xla_enabled
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.is_complex
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: is_complex(input) -> (bool)

Returns True if the data type of :attr:`input` is a complex data type i.e.,
one of ``torch.complex64``, and ``torch.complex128``.

Args:
    input (Tensor): the input tens...

#### torch.is_conj
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: is_conj(input) -> (bool)

Returns True if the :attr:`input` is a conjugated tensor, i.e. its conjugate bit is set to `True`.

Args:
    input (Tensor): the input tensor....

#### torch.is_floating_point
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: is_floating_point(input) -> (bool)

Returns True if the data type of :attr:`input` is a floating point data type i.e.,
one of ``torch.float64``, ``torch.float32``, ``torch.float16``, and ``torch.bfloa...

#### torch.is_inference
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: is_inference(input) -> (bool)

Returns True if :attr:`input` is an inference tensor.

A non-view tensor is an inference tensor if and only if it was
allocated during inference mode. A view tensor is a...

#### torch.is_nonzero
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: is_nonzero(input) -> (bool)

Returns True if the :attr:`input` is a single element tensor which is not equal to zero
after type conversions.
i.e. not equal to ``torch.tensor([0.])`` or ``torch.tensor(...

#### torch.isclose
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

Returns a new tensor with boolean elements representing if each element of
:attr:`input` is "close" to the corresponding eleme...

#### torch.isfinite
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: isfinite(input) -> Tensor

Returns a new tensor with boolean elements representing if each element is `finite` or not.

Real values are finite when they are not NaN, negative infinity, or infinity.
Co...

#### torch.isnan
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: isnan(input) -> Tensor

Returns a new tensor with boolean elements representing if each element of :attr:`input`
is NaN or not. Complex values are considered NaN when either their real
and/or imaginar...

#### torch.isneginf
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: isneginf(input, *, out=None) -> Tensor
Tests if each element of :attr:`input` is negative infinity or not.

Args:
  input (Tensor): the input tensor.

Keyword args:
  out (Tensor, optional): the outpu...

#### torch.isreal
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: isreal(input) -> Tensor

Returns a new tensor with boolean elements representing if each element of :attr:`input` is real-valued or not.
All real-valued types are considered real. Complex values are c...

#### torch.istft
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: istft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False, onesided=None, length=None, return_complex=False) -> Tensor:

Inverse short time Fourier Transform. Th...

#### torch.kaiser_window
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: kaiser_window(window_length, periodic=True, beta=12.0, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Computes the Kaiser window with window length :attr:`window_len...

#### torch.kron
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: kron(input, other, *, out=None) -> Tensor

Computes the Kronecker product, denoted by :math:`\otimes`, of :attr:`input` and :attr:`other`.

If :attr:`input` is a :math:`(a_0 \times a_1 \times \dots \t...

#### torch.kthvalue
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: kthvalue(input, k, dim=None, keepdim=False, *, out=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` is the :attr:`k` th
smallest element of each row of the :a...

#### torch.lcm
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: lcm(input, other, *, out=None) -> Tensor

Computes the element-wise least common multiple (LCM) of :attr:`input` and :attr:`other`.

Both :attr:`input` and :attr:`other` must have integer types.

.. n...

#### torch.le
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: le(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} \leq \text{other}` element-wise.


The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcastin...

#### torch.lerp
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: lerp(input, end, weight, *, out=None)

Does a linear interpolation of two tensors :attr:`start` (given by :attr:`input`) and :attr:`end` based
on a scalar or tensor :attr:`weight` and returns the resu...

#### torch.less
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: less(input, other, *, out=None) -> Tensor

Alias for :func:`torch.lt`....

#### torch.less_equal
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: less_equal(input, other, *, out=None) -> Tensor

Alias for :func:`torch.le`....

#### torch.lgamma
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: lgamma(input, *, out=None) -> Tensor

Computes the natural logarithm of the absolute value of the gamma function on :attr:`input`.

.. math::
    \text{out}_{i} = \ln |\Gamma(\text{input}_{i})|

Args:...

#### torch.load
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: load(f, map_location=None, pickle_module=pickle, *, weights_only=True, mmap=None, **pickle_load_args)

Loads an object saved with :func:`torch.save` from a file.

:func:`torch.load` uses Python's unpi...

#### torch.lobpcg
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Find the k largest (or smallest) eigenvalues and the corresponding
eigenvectors of a symmetric positive definite generalized
eigenvalue problem using matrix-free LOBPCG methods.

This function is a fr...

#### torch.lt
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: lt(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} < \text{other}` element-wise.


The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-s...

#### torch.lu
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Computes the LU factorization of a matrix or batches of matrices
:attr:`A`. Returns a tuple containing the LU factorization and
pivots of :attr:`A`.  Pivoting is done if :attr:`pivot` is set to
``True...

#### torch.lu_solve
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: lu_solve(b, LU_data, LU_pivots, *, out=None) -> Tensor

Returns the LU solve of the linear system :math:`Ax = b` using the partially pivoted
LU factorization of A from :func:`~linalg.lu_factor`.

This...

#### torch.lu_unpack
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True, *, out=None) -> (Tensor, Tensor, Tensor)

Unpacks the LU decomposition returned by :func:`~linalg.lu_factor` into the `P, L, U` matr...

#### torch.manual_seed
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the seed for generating random numbers on all devices. Returns a
`torch.Generator` object.

Args:
    seed (int): The desired seed. Value must be within the inclusive range
        `[-0x8000_0000...

#### torch.masked_select
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: masked_select(input, mask, *, out=None) -> Tensor

Returns a new 1-D tensor which indexes the :attr:`input` tensor according to
the boolean mask :attr:`mask` which is a `BoolTensor`.

The shapes of th...

#### torch.matmul
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: matmul(input, other, *, out=None) -> Tensor

Matrix product of two tensors.

The behavior depends on the dimensionality of the tensors as follows:

- If both tensors are 1-dimensional, the dot product...

#### torch.matrix_power
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: matrix_power(input, n, *, out=None) -> Tensor

Alias for :func:`torch.linalg.matrix_power`...

#### torch.max
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: max(input) -> Tensor

Returns the maximum value of all elements in the ``input`` tensor.

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ ...

#### torch.maximum
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: maximum(input, other, *, out=None) -> Tensor

Computes the element-wise maximum of :attr:`input` and :attr:`other`.

.. note::
    If one of the elements being compared is a NaN, then that element is ...

#### torch.mean
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: mean(input, *, dtype=None) -> Tensor

.. note::
    If the `input` tensor is empty, ``torch.mean()`` returns ``nan``.
    This behavior is consistent with NumPy and follows the definition
    that the...

#### torch.median
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: median(input) -> Tensor

Returns the median of the values in :attr:`input`.

.. note::
    The median is not unique for :attr:`input` tensors with an even number
    of elements. In this case the lowe...

#### torch.meshgrid
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates grids of coordinates specified by the 1D inputs in `attr`:tensors.

This is helpful when you want to visualize data over some
range of inputs. See below for a plotting example.

Given :math:`N...

#### torch.min
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: min(input) -> Tensor

Returns the minimum value of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor...

#### torch.minimum
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: minimum(input, other, *, out=None) -> Tensor

Computes the element-wise minimum of :attr:`input` and :attr:`other`.

.. note::
    If one of the elements being compared is a NaN, then that element is ...

#### torch.mm
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: mm(input, mat2, out_dtype=None, *, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`input` and :attr:`mat2`.

If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`mat2...

#### torch.mode
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: mode(input, dim=-1, keepdim=False, *, out=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` is the mode
value of each row of the :attr:`input` tensor in the gi...

#### torch.moveaxis
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: moveaxis(input, source, destination) -> Tensor

Alias for :func:`torch.movedim`.

This function is equivalent to NumPy's moveaxis function.

Examples::

    >>> t = torch.randn(3,2,1)
    >>> t
    te...

#### torch.movedim
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: movedim(input, source, destination) -> Tensor

Moves the dimension(s) of :attr:`input` at the position(s) in :attr:`source`
to the position(s) in :attr:`destination`.

Other dimensions of :attr:`input...

#### torch.msort
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: msort(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Sorts the elements of the :attr:`input` tensor along its first dimension
in ascending order by value.

.. note:: `torch.msort(t)` is equivalen...

#### torch.mul
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: mul(input, other, *, out=None) -> Tensor

Multiplies :attr:`input` by :attr:`other`.


.. math::
    \text{out}_i = \text{input}_i \times \text{other}_i


Supports :ref:`broadcasting to a common shape...

#### torch.multinomial
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: multinomial(input, num_samples, replacement=False, *, generator=None, out=None) -> LongTensor

Returns a tensor where each row contains :attr:`num_samples` indices sampled
from the multinomial (a stri...

#### torch.mv
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: mv(input, vec, *, out=None) -> Tensor

Performs a matrix-vector product of the matrix :attr:`input` and the vector
:attr:`vec`.

If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-D...

#### torch.mvlgamma
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: mvlgamma(input, p, *, out=None) -> Tensor

Alias for :func:`torch.special.multigammaln`....

#### torch.nan_to_num
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None) -> Tensor

Replaces :literal:`NaN`, positive infinity, and negative infinity values in :attr:`input`
with the values specified by :att...

#### torch.nanmean
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: nanmean(input, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor

Computes the mean of all `non-NaN` elements along the specified dimensions.
Input must be floating point or complex.

This f...

#### torch.nanmedian
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: nanmedian(input) -> Tensor

Returns the median of the values in :attr:`input`, ignoring ``NaN`` values.

This function is identical to :func:`torch.median` when there are no ``NaN`` values in :attr:`i...

#### torch.nanquantile
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: nanquantile(input, q, dim=None, keepdim=False, *, interpolation='linear', out=None) -> Tensor

This is a variant of :func:`torch.quantile` that "ignores" ``NaN`` values,
computing the quantiles :attr:...

#### torch.nansum
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: nansum(input, *, dtype=None) -> Tensor

Returns the sum of all elements, treating Not a Numbers (NaNs) as zero.

Args:
    input (Tensor): the input tensor.

Keyword args:
    dtype (:class:`torch.dty...

#### torch.narrow
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: narrow(input, dim, start, length) -> Tensor

Returns a new tensor that is a narrowed version of :attr:`input` tensor. The
dimension :attr:`dim` is input from :attr:`start` to ``start + length``. The
r...

#### torch.narrow_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: narrow_copy(input, dim, start, length, *, out=None) -> Tensor

Same as :meth:`Tensor.narrow` except this returns a copy rather
than shared storage. This is primarily for sparse tensors, which
do not h...

#### torch.ne
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ne(input, other, *, out=None) -> Tensor

Computes :math:`\text{input} \neq \text{other}` element-wise.


The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcastin...

#### torch.neg
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: neg(input, *, out=None) -> Tensor

Returns a new tensor with the negative of the elements of :attr:`input`.

.. math::
    \text{out} = -1 \times \text{input}

Args:
    input (Tensor): the input tens...

#### torch.negative
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: negative(input, *, out=None) -> Tensor

Alias for :func:`torch.neg`...

#### torch.nextafter
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: nextafter(input, other, *, out=None) -> Tensor

Return the next floating-point value after :attr:`input` towards :attr:`other`, elementwise.

The shapes of ``input`` and ``other`` must be
:ref:`broadc...

#### torch.no_grad
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Context-manager that disables gradient calculation.

Disabling gradient calculation is useful for inference, when you are sure
that you will not call :meth:`Tensor.backward()`. It will reduce memory
c...

#### torch.nonzero
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: nonzero(input, *, out=None, as_tuple=False) -> LongTensor or tuple of LongTensors

.. note::
    :func:`torch.nonzero(..., as_tuple=False) <torch.nonzero>` (default) returns a
    2-D tensor where eac...

#### torch.norm
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the matrix norm or vector norm of a given tensor.

.. warning::

    torch.norm is deprecated and may be removed in a future PyTorch release.
    Its documentation and behavior may be incorrec...

#### torch.not_equal
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: not_equal(input, other, *, out=None) -> Tensor

Alias for :func:`torch.ne`....

#### torch.numel
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: numel(input: Tensor) -> int

Returns the total number of elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(1, 2, 3, 4, 5)
    >>> t...

#### torch.orgqr
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: orgqr(input, tau) -> Tensor

Alias for :func:`torch.linalg.householder_product`....

#### torch.ormqr
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ormqr(input, tau, other, left=True, transpose=False, *, out=None) -> Tensor

Computes the matrix-matrix multiplication of a product of Householder matrices with a general matrix.

Multiplies a :math:`...

#### torch.outer
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: outer(input, vec2, *, out=None) -> Tensor

Outer product of :attr:`input` and :attr:`vec2`.
If :attr:`input` is a vector of size :math:`n` and :attr:`vec2` is a vector of
size :math:`m`, then :attr:`o...

#### torch.parse_ir
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: parse_ir(input: str, parse_tensor_constants: bool = False) -> torch::jit::Graph...

#### torch.pca_lowrank
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs linear Principal Component Analysis (PCA) on a low-rank
matrix, batches of such matrices, or sparse matrix.

This function returns a namedtuple ``(U, S, V)`` which is the
nearly optimal appro...

#### torch.pdist
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: pdist(input, p=2) -> Tensor

Computes the p-norm distance between every pair of row vectors in the input.
This is identical to the upper triangular portion, excluding the diagonal, of
`torch.norm(inpu...

#### torch.permute
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: permute(input, dims) -> Tensor

Returns a view of the original tensor :attr:`input` with its dimensions permuted.

Args:
    input (Tensor): the input tensor.
    dims (tuple of int): The desired orde...

#### torch.permute_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.permute`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.pinverse
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: pinverse(input, rcond=1e-15) -> Tensor

Alias for :func:`torch.linalg.pinv`...

#### torch.pixel_shuffle
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: pixel_shuffle(input, upscale_factor) -> Tensor

Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)` to a
tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is the :at...

#### torch.pixel_unshuffle
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: pixel_unshuffle(input, downscale_factor) -> Tensor

Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements in a
tensor of shape :math:`(*, C, H \times r, W \times r)` to a tens...

#### torch.poisson
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: poisson(input, generator=None) -> Tensor

Returns a tensor of the same size as :attr:`input` with each element
sampled from a Poisson distribution with rate parameter given by the corresponding
elemen...

#### torch.polar
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: polar(abs, angle, *, out=None) -> Tensor

Constructs a complex tensor whose elements are Cartesian coordinates
corresponding to the polar coordinates with absolute value :attr:`abs` and angle
:attr:`a...

#### torch.polygamma
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: polygamma(n, input, *, out=None) -> Tensor

Alias for :func:`torch.special.polygamma`....

#### torch.positive
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: positive(input) -> Tensor

Returns :attr:`input`.
Throws a runtime error if :attr:`input` is a bool tensor.

Args:
    input (Tensor): the input tensor.

Example::

    >>> t = torch.randn(5)
    >>> ...

#### torch.pow
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: pow(input, exponent, *, out=None) -> Tensor

Takes the power of each element in :attr:`input` with :attr:`exponent` and
returns a tensor with the result.

:attr:`exponent` can be either a single ``flo...

#### torch.prod
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: prod(input: Tensor, *, dtype: Optional[_dtype]) -> Tensor

Returns the product of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Keyword args:
    dtype (:clas...

#### torch.qr
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: qr(input: Tensor, some: bool = True, *, out: Union[Tensor, Tuple[Tensor, ...], List[Tensor], None]) -> (Tensor, Tensor)

Computes the QR decomposition of a matrix or a batch of matrices :attr:`input`,...

#### torch.quantile
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: quantile(input, q, dim=None, keepdim=False, *, interpolation='linear', out=None) -> Tensor

Computes the q-th quantiles of each row of the :attr:`input` tensor along the dimension :attr:`dim`.

To com...

#### torch.quantized_batch_norm
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: quantized_batch_norm(input, weight=None, bias=None, mean, var, eps, output_scale, output_zero_point) -> Tensor

Applies batch normalization on a 4D (NCHW) quantized tensor.

.. math::

        y = \fr...

#### torch.rad2deg
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: rad2deg(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with each of the elements of :attr:`input`
converted from angles in radians to degrees.

Args:
    input (Tensor): the ...

#### torch.range
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: range(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a 1-D tensor of size :math:`\left\lfloor \frac{\text{end} - \text{start}...

#### torch.ravel
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ravel(input) -> Tensor

Return a contiguous flattened tensor. A copy is made only if needed.

Args:
    input (Tensor): the input tensor.

Example::

    >>> t = torch.tensor([[[1, 2],
    ...        ...

#### torch.real
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: real(input) -> Tensor

Returns a new tensor containing real values of the :attr:`self` tensor.
The returned tensor and :attr:`self` share the same underlying storage.

Args:
    input (Tensor): the in...

#### torch.reciprocal
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: reciprocal(input, *, out=None) -> Tensor

Returns a new tensor with the reciprocal of the elements of :attr:`input`

.. math::
    \text{out}_{i} = \frac{1}{\text{input}_{i}}

.. note::
    Unlike Num...

#### torch.remainder
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: remainder(input, other, *, out=None) -> Tensor

Computes
`Python's modulus operation <https://docs.python.org/3/reference/expressions.html#binary-arithmetic-operations>`_
entrywise.  The result has th...

#### torch.renorm
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: renorm(input, p, dim, maxnorm, *, out=None) -> Tensor

Returns a tensor where each sub-tensor of :attr:`input` along dimension
:attr:`dim` is normalized such that the `p`-norm of the sub-tensor is low...

#### torch.repeat_interleave
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: repeat_interleave(input, repeats, dim=None, *, output_size=None) -> Tensor

Repeat elements of a tensor.

.. warning::

    This is different from :meth:`torch.Tensor.repeat` but similar to ``numpy.re...

#### torch.reshape
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: reshape(input, shape) -> Tensor

Returns a tensor with the same data and number of elements as :attr:`input`,
but with the specified shape. When possible, the returned tensor will be a view
of :attr:`...

#### torch.resolve_conj
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: resolve_conj(input) -> Tensor

Returns a new tensor with materialized conjugation if :attr:`input`'s conjugate bit is set to `True`,
else returns :attr:`input`. The output tensor will always have its ...

#### torch.resolve_neg
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: resolve_neg(input) -> Tensor

Returns a new tensor with materialized negation if :attr:`input`'s negative bit is set to `True`,
else returns :attr:`input`. The output tensor will always have its negat...

#### torch.result_type
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: result_type(tensor1, tensor2) -> dtype

Returns the :class:`torch.dtype` that would result from performing an arithmetic
operation on the provided input tensors. See type promotion :ref:`documentation...

#### torch.roll
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: roll(input, shifts, dims=None) -> Tensor

Roll the tensor :attr:`input` along the given dimension(s). Elements that are
shifted beyond the last position are re-introduced at the first position. If
:at...

#### torch.rot90
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: rot90(input, k=1, dims=(0, 1)) -> Tensor

Rotate an n-D tensor by 90 degrees in the plane specified by dims axis.
Rotation direction is from the first towards the second axis if k > 0, and from the se...

#### torch.round
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: round(input, *, decimals=0, out=None) -> Tensor

Rounds elements of :attr:`input` to the nearest integer.

For integer inputs, follows the array-api convention of returning a
copy of the input tensor....

#### torch.row_stack
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: row_stack(tensors, *, out=None) -> Tensor

Alias of :func:`torch.vstack`....

#### torch.save
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True)

Saves an object to a disk file.

See also: :ref:`saving-loading-tensors`

Args:
    obj: saved object
    f:...

#### torch.scatter
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: scatter(input, dim, index, src) -> Tensor

Out-of-place version of :meth:`torch.Tensor.scatter_`...

#### torch.scatter_add
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: scatter_add(input, dim, index, src) -> Tensor

Out-of-place version of :meth:`torch.Tensor.scatter_add_`...

#### torch.scatter_reduce
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: scatter_reduce(input, dim, index, src, reduce, *, include_self=True) -> Tensor

Out-of-place version of :meth:`torch.Tensor.scatter_reduce_`...

#### torch.searchsorted
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: searchsorted(sorted_sequence, values, *, out_int32=False, right=False, side=None, out=None, sorter=None) -> Tensor

Find the indices from the *innermost* dimension of :attr:`sorted_sequence` such that...

#### torch.seed
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the seed for generating random numbers to a non-deterministic
random number on all devices. Returns a 64 bit number used to seed the RNG....

#### torch.select
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: select(input, dim, index) -> Tensor

Slices the :attr:`input` tensor along the selected dimension at the given index.
This function returns a view of the original tensor with the given dimension remov...

#### torch.select_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.select`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.select_scatter
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: select_scatter(input, src, dim, index) -> Tensor

Embeds the values of the :attr:`src` tensor into :attr:`input` at the given index.
This function returns a tensor with fresh storage; it does not crea...

#### torch.selu_
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: selu_(input) -> Tensor

In-place version of :func:`~selu`....

#### torch.set_autocast_xla_dtype
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.set_autocast_xla_enabled
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.set_default_dtype
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the default floating point dtype to :attr:`d`. Supports floating point dtype
as inputs. Other dtypes will cause torch to raise an exception.

When PyTorch is initialized its default floating poin...

#### torch.set_float32_matmul_precision
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the internal precision of float32 matrix multiplications.

Running float32 matrix multiplications in lower precision may significantly increase
performance, and in some programs the loss of preci...

#### torch.set_grad_enabled
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Context-manager that sets gradient calculation on or off.

``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
It can be used as a context-manager or as a function.

...

#### torch.set_num_interop_threads
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: set_num_interop_threads(int)

Sets the number of threads used for interop parallelism
(e.g. in JIT interpreter) on CPU.

.. warning::
    Can only be called once and before any inter-op parallel work
...

#### torch.set_num_threads
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: set_num_threads(int)

Sets the number of threads used for intraop parallelism on CPU.

.. warning::
    To ensure that the correct number of threads is used, set_num_threads
    must be called before ...

#### torch.set_printoptions
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Set options for printing. Items shamelessly taken from NumPy

Args:
    precision: Number of digits of precision for floating point output
        (default = 4).
    threshold: Total number of array e...

#### torch.set_rng_state
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the random number generator state.

.. note:: This function only works for CPU. For CUDA, please use
    :func:`torch.manual_seed`, which works for both CPU and CUDA.

Args:
    new_state (torch....

#### torch.sgn
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sgn(input, *, out=None) -> Tensor

This function is an extension of torch.sign() to complex tensors.
It computes a new tensor whose elements have
the same angles as the corresponding elements of :attr...

#### torch.sign
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sign(input, *, out=None) -> Tensor

Returns a new tensor with the signs of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \operatorname{sgn}(\text{input}_{i})

Args:
    input (Tensor)...

#### torch.signbit
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: signbit(input, *, out=None) -> Tensor

Tests if each element of :attr:`input` has its sign bit set or not.

Args:
  input (Tensor): the input tensor.

Keyword args:
  out (Tensor, optional): the outpu...

#### torch.slice_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.slice`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.slice_scatter
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: slice_scatter(input, src, dim=0, start=None, end=None, step=1) -> Tensor

Embeds the values of the :attr:`src` tensor into :attr:`input` at the given
dimension.
This function returns a tensor with fre...

#### torch.smm
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: smm(input, mat) -> Tensor

Performs a matrix multiplication of the sparse matrix :attr:`input`
with the dense matrix :attr:`mat`.

Args:
    input (Tensor): a sparse matrix to be matrix multiplied
   ...

#### torch.sort
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sort(input, dim=-1, descending=False, stable=False, *, out=None) -> (Tensor, LongTensor)

Sorts the elements of the :attr:`input` tensor along a given dimension
in ascending order by value.

If :attr:...

#### torch.split
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Splits the tensor into chunks. Each chunk is a view of the original tensor.

If :attr:`split_size_or_sections` is an integer type, then :attr:`tensor` will
be split into equally sized chunks (if possi...

#### torch.split_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.split`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.split_with_sizes_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.split_with_sizes`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.square
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: square(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the square of the elements of :attr:`input`.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (T...

#### torch.squeeze
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: squeeze(input: Tensor, dim: Optional[Union[int, List[int]]]) -> Tensor

Returns a tensor with all specified dimensions of :attr:`input` of size `1` removed.

For example, if `input` is of shape:
:math...

#### torch.squeeze_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.squeeze`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.sspaddmm
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sspaddmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) -> Tensor

Matrix multiplies a sparse tensor :attr:`mat1` with a dense tensor
:attr:`mat2`, then adds the sparse tensor :attr:`input` to the r...

#### torch.stack
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: stack(tensors, dim=0, *, out=None) -> Tensor

Concatenates a sequence of tensors along a new dimension.

All tensors need to be of the same size.

.. seealso::

    :func:`torch.cat` concatenates the ...

#### torch.std
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: std(input, dim=None, *, correction=1, keepdim=False, out=None) -> Tensor

Calculates the standard deviation over the dimensions specified by :attr:`dim`.
:attr:`dim` can be a single dimension, list of...

#### torch.std_mean
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: std_mean(input, dim=None, *, correction=1, keepdim=False, out=None) -> (Tensor, Tensor)

Calculates the standard deviation and mean over the dimensions specified by
:attr:`dim`. :attr:`dim` can be a s...

#### torch.stft
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Short-time Fourier transform (STFT).

.. warning::
    From version 1.8.0, :attr:`return_complex` must always be given
    explicitly for real inputs and `return_complex=False` has been
    deprecated...

#### torch.sub
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sub(input, other, *, alpha=1, out=None) -> Tensor

Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`.

.. math::
    \text{{out}}_i = \text{{input}}_i - \text{{alpha}} \times \text{...

#### torch.subtract
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: subtract(input, other, *, alpha=1, out=None) -> Tensor

Alias for :func:`torch.sub`....

#### torch.sum
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sum(input, *, dtype=None) -> Tensor

Returns the sum of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional)...

#### torch.svd
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: svd(input, some=True, compute_uv=True, *, out=None) -> (Tensor, Tensor, Tensor)

Computes the singular value decomposition of either a matrix or batch of
matrices :attr:`input`. The singular value dec...

#### torch.svd_lowrank
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return the singular value decomposition ``(U, S, V)`` of a matrix,
batches of matrices, or a sparse matrix :math:`A` such that
:math:`A \approx U \operatorname{diag}(S) V^{\text{H}}`. In case :math:`M...

#### torch.swapaxes
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: swapaxes(input, axis0, axis1) -> Tensor

Alias for :func:`torch.transpose`.

This function is equivalent to NumPy's swapaxes function.

Examples::

    >>> x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]...

#### torch.swapdims
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: swapdims(input, dim0, dim1) -> Tensor

Alias for :func:`torch.transpose`.

This function is equivalent to NumPy's swapaxes function.

Examples::

    >>> x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]]...

#### torch.t
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: t(input) -> Tensor

Expects :attr:`input` to be <= 2-D tensor and transposes dimensions 0
and 1.

0-D and 1-D tensors are returned as is. When input is a 2-D tensor this
is equivalent to ``transpose(i...

#### torch.t_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.t`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.take
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: take(input, index) -> Tensor

Returns a new tensor with the elements of :attr:`input` at the given indices.
The input tensor is treated as if it were viewed as a 1-D tensor. The result
takes the same ...

#### torch.take_along_dim
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: take_along_dim(input, indices, dim=None, *, out=None) -> Tensor

Selects values from :attr:`input` at the 1-dimensional indices from :attr:`indices` along the given :attr:`dim`.

If :attr:`dim` is Non...

#### torch.threshold_
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: threshold_(input, threshold, value) -> Tensor

In-place version of :func:`~threshold`....

#### torch.tile
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: tile(input, dims) -> Tensor

Constructs a tensor by repeating the elements of :attr:`input`.
The :attr:`dims` argument specifies the number of repetitions
in each dimension.

If :attr:`dims` specifies...

#### torch.to_dlpack
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: to_dlpack(tensor) -> PyCapsule

Returns an opaque object (a "DLPack capsule") representing the tensor.

.. note::
  ``to_dlpack`` is a legacy DLPack interface. The capsule it returns
  cannot be used ...

#### torch.topk
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: topk(input, k, dim=None, largest=True, sorted=True, *, out=None) -> (Tensor, LongTensor)

Returns the :attr:`k` largest elements of the given :attr:`input` tensor along
a given dimension.

If :attr:`d...

#### torch.trace
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: trace(input) -> Tensor

Returns the sum of the elements of the diagonal of the input 2-D matrix.

Example::

    >>> x = torch.arange(1., 10.).view(3, 3)
    >>> x
    tensor([[ 1.,  2.,  3.],
       ...

#### torch.transpose
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: transpose(input, dim0, dim1) -> Tensor

Returns a tensor that is a transposed version of :attr:`input`.
The given dimensions :attr:`dim0` and :attr:`dim1` are swapped.

If :attr:`input` is a strided t...

#### torch.transpose_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.transpose`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.trapezoid
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: trapezoid(y, x=None, *, dx=None, dim=-1) -> Tensor

Computes the `trapezoidal rule <https://en.wikipedia.org/wiki/Trapezoidal_rule>`_ along
:attr:`dim`. By default the spacing between elements is assu...

#### torch.trapz
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: trapz(y, x, *, dim=-1) -> Tensor

Alias for :func:`torch.trapezoid`....

#### torch.triangular_solve
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: triangular_solve(b, A, upper=True, transpose=False, unitriangular=False, *, out=None) -> (Tensor, Tensor)

Solves a system of equations with a square upper or lower triangular invertible matrix :math:...

#### torch.tril
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: tril(input, diagonal=0, *, out=None) -> Tensor

Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices
:attr:`input`, the other elements of the result tensor :attr:`out` are...

#### torch.tril_indices
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: tril_indices(row, col, offset=0, *, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor

Returns the indices of the lower triangular part of a :attr:`row`-by-
:attr:`col` matrix in a 2-by-...

#### torch.triu
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: triu(input, diagonal=0, *, out=None) -> Tensor

Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices
:attr:`input`, the other elements of the result tensor :attr:`out` are s...

#### torch.triu_indices
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: triu_indices(row, col, offset=0, *, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor

Returns the indices of the upper triangular part of a :attr:`row` by
:attr:`col` matrix in a 2-by-N...

#### torch.true_divide
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: true_divide(dividend, divisor, *, out) -> Tensor

Alias for :func:`torch.div` with ``rounding_mode=None``....

#### torch.trunc
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: trunc(input, *, out=None) -> Tensor

Returns a new tensor with the truncated integer values of
the elements of :attr:`input`.

For integer inputs, follows the array-api convention of returning a
copy ...

#### torch.typename
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: String representation of the type of an object.

This function returns a fully qualified string representation of an object's type.
Args:
    obj (object): The object whose type to represent
Returns:
...

#### torch.unbind
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: unbind(input, dim=0) -> seq

Removes a tensor dimension.

Returns a tuple of all slices along a given dimension, already without it.

Arguments:
    input (Tensor): the tensor to unbind
    dim (int):...

#### torch.unbind_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.unbind`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.unflatten
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: unflatten(input, dim, sizes) -> Tensor

Expands a dimension of the input tensor over multiple dimensions.

.. seealso::

    :func:`torch.flatten` the inverse of this function. It coalesces several di...

#### torch.unfold_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.unfold`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.unique
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> tuple[Tensor, Tensor, Tensor]

Returns the unique elements of the input tensor.

.. note:: This function is different...

#### torch.unique_consecutive
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Eliminates all but the first element from every consecutive group of equivalent elements.

.. note:: This function is different from :func:`torch.unique` in the sense that this function
    only elimi...

#### torch.unravel_index
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Converts a tensor of flat indices into a tuple of coordinate tensors that
index into an arbitrary tensor of the specified shape.

Args:
    indices (Tensor): An integer tensor containing indices into ...

#### torch.unsafe_chunk
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: unsafe_chunk(input, chunks, dim=0) -> List of Tensors

Works like :func:`torch.chunk` but without enforcing the autograd restrictions
on inplace modification of the outputs.

.. warning::
    This fun...

#### torch.unsafe_split
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: unsafe_split(tensor, split_size_or_sections, dim=0) -> List of Tensors

Works like :func:`torch.split` but without enforcing the autograd restrictions
on inplace modification of the outputs.

.. warni...

#### torch.unsqueeze
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: unsqueeze(input, dim) -> Tensor

Returns a new tensor with a dimension of size one inserted at the
specified position.

The returned tensor shares the same underlying data with this tensor.

A :attr:`...

#### torch.unsqueeze_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.unsqueeze`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.use_deterministic_algorithms
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets whether PyTorch operations must use "deterministic"
algorithms. That is, algorithms which, given the same input, and when
run on the same software and hardware, always produce the same output.
Wh...

#### torch.values_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.values`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.vander
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: vander(x, N=None, increasing=False) -> Tensor

Generates a Vandermonde matrix.

The columns of the output matrix are elementwise powers of the input vector :math:`x^{(N-1)}, x^{(N-2)}, ..., x^0`.
If i...

#### torch.var
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: var(input, dim=None, *, correction=1, keepdim=False, out=None) -> Tensor

Calculates the variance over the dimensions specified by :attr:`dim`. :attr:`dim`
can be a single dimension, list of dimension...

#### torch.var_mean
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: var_mean(input, dim=None, *, correction=1, keepdim=False, out=None) -> (Tensor, Tensor)

Calculates the variance and mean over the dimensions specified by :attr:`dim`.
:attr:`dim` can be a single dime...

#### torch.vdot
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: vdot(input, other, *, out=None) -> Tensor

Computes the dot product of two 1D vectors along a dimension.

In symbols, this function computes

.. math::

    \sum_{i=1}^n \overline{x_i}y_i.

where :mat...

#### torch.view_as_complex
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: view_as_complex(input) -> Tensor

Returns a view of :attr:`input` as a complex tensor. For an input complex
tensor of :attr:`size` :math:`m1, m2, \dots, mi, 2`, this function returns a
new complex ten...

#### torch.view_as_complex_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.view_as_complex`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.view_as_real
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: view_as_real(input) -> Tensor

Returns a view of :attr:`input` as a real tensor. For an input complex tensor of
:attr:`size` :math:`m1, m2, \dots, mi`, this function returns a new
real tensor of size ...

#### torch.view_as_real_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.view_as_real`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.view_copy
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.view`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.vmap
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: vmap is the vectorizing map; ``vmap(func)`` returns a new function that
maps ``func`` over some dimension of the inputs. Semantically, vmap
pushes the map into PyTorch operations called by ``func``, e...

#### torch.vsplit
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: vsplit(input, indices_or_sections) -> List of Tensors

Splits :attr:`input`, a tensor with two or more dimensions, into multiple tensors
vertically according to :attr:`indices_or_sections`. Each split...

#### torch.vstack
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: vstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence vertically (row wise).

This is equivalent to concatenation along the first axis after all 1-D tensors have been reshaped by :func:`to...

#### torch.where
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: where(condition, input, other, *, out=None) -> Tensor

Return a tensor of elements selected from either :attr:`input` or :attr:`other`, depending on :attr:`condition`.

The operation is defined as:

....

#### torch.while_loop
- **Category**: TORCH_CORE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Run body_fn(*carried_inputs) while cond_fn(*carried_inputs) returns a True scalar tensor. Returns the output of body_fn or
initial carried_inputs.

.. warning::
    `torch.while_loop` is a prototype f...

#### torch.arange
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a 1-D tensor of size :math:`\left\lceil \frac{\text{end} - \text{start}...

#### torch.empty
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format) -> Tensor

Returns a tensor filled with uninitial...

#### torch.empty_like
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns an uninitialized tensor with the same size as :attr:`input`.
``t...

#### torch.empty_permuted
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: empty_permuted(size, physical_layout, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) -> Tensor

Creates an uninitialized, non-overlapping and dense tensor with the
spe...

#### torch.empty_strided
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: empty_strided(size, stride, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) -> Tensor

Creates a tensor with the specified :attr:`size` and :attr:`stride` and filled wi...

#### torch.eye
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

Args:
    n (int): the n...

#### torch.linspace
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Creates a one-dimensional tensor of size :attr:`steps` whose values are evenly
s...

#### torch.ones
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with the scalar value `1`, with the shape defined
by the variable argume...

#### torch.ones_like
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor filled with the scalar value `1`, with the same size as
...

#### torch.zeros
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with the scalar value `0`, with the shape defined
by the variable argum...

#### torch.zeros_like
- **Category**: TORCH_TENSOR_CREATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor filled with the scalar value `0`, with the same size as...

#### torch.get_autocast_cpu_dtype
- **Category**: TORCH_DEVICE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.is_autocast_cpu_enabled
- **Category**: TORCH_DEVICE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.set_autocast_cpu_dtype
- **Category**: TORCH_DEVICE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.set_autocast_cpu_enabled
- **Category**: TORCH_DEVICE
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.abs
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: abs(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Computes the absolute value of each element in :attr:`input`.

.. math::
    \text{out}_{i} = |\text{input}_{i}|

Args:
    input (Tensor): the ...

#### torch.absolute
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: absolute(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.abs`...

#### torch.acos
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: acos(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Computes the inverse cosine of each element in :attr:`input`.

.. math::
    \text{out}_{i} = \cos^{-1}(\text{input}_{i})

Args:
    input (Ten...

#### torch.acosh
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: acosh(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the inverse hyperbolic cosine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \cosh^{-1}(\text{inp...

#### torch.arccos
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: arccos(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.acos`....

#### torch.arccosh
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: arccosh(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.acosh`....

#### torch.arcsin
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: arcsin(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.asin`....

#### torch.arcsinh
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: arcsinh(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.asinh`....

#### torch.arctan
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: arctan(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.atan`....

#### torch.arctan2
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: arctan2(input: Tensor, other: Tensor, *, out: Optional[Tensor]) -> Tensor
Alias for :func:`torch.atan2`....

#### torch.arctanh
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: arctanh(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.atanh`....

#### torch.asin
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: asin(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the arcsine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sin^{-1}(\text{input}_{i})

Args:
    ...

#### torch.asinh
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: asinh(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the inverse hyperbolic sine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sinh^{-1}(\text{input...

#### torch.atan
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: atan(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the arctangent of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \tan^{-1}(\text{input}_{i})

Args:
 ...

#### torch.atan2
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: atan2(input: Tensor, other: Tensor, *, out: Optional[Tensor]) -> Tensor

Element-wise arctangent of :math:`\text{input}_{i} / \text{other}_{i}`
with consideration of the quadrant. Returns a new tensor...

#### torch.atanh
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: atanh(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the inverse hyperbolic tangent of the elements of :attr:`input`.

Note:
    The domain of the inverse hyperbolic tan...

#### torch.ceil
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ceil(input, *, out=None) -> Tensor

Returns a new tensor with the ceil of the elements of :attr:`input`,
the smallest integer greater than or equal to each element.

For integer inputs, follows the ar...

#### torch.cos
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cos(input, *, out=None) -> Tensor

Returns a new tensor with the cosine  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \cos(\text{input}_{i})

Args:
    input (Tensor): the input t...

#### torch.cosh
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cosh(input, *, out=None) -> Tensor

Returns a new tensor with the hyperbolic cosine  of the elements of
:attr:`input`.

.. math::
    \text{out}_{i} = \cosh(\text{input}_{i})

Args:
    input (Tensor)...

#### torch.cosine_similarity
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cosine_similarity(x1, x2, dim=1, eps=1e-8) -> Tensor

Returns cosine similarity between ``x1`` and ``x2``, computed along dim. ``x1`` and ``x2`` must be broadcastable
to a common shape. ``dim`` refers...

#### torch.exp
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: exp(input, *, out=None) -> Tensor

Returns a new tensor with the exponential of the elements
of the input tensor :attr:`input`.

.. math::
    y_{i} = e^{x_{i}}

Args:
    input (Tensor): the input te...

#### torch.exp2
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: exp2(input, *, out=None) -> Tensor

Alias for :func:`torch.special.exp2`....

#### torch.expand_copy
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs the same operation as :func:`torch.Tensor.expand`, but all output tensors
are freshly created instead of aliasing the input....

#### torch.expm1
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: expm1(input, *, out=None) -> Tensor

Alias for :func:`torch.special.expm1`....

#### torch.floor
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: floor(input, *, out=None) -> Tensor

Returns a new tensor with the floor of the elements of :attr:`input`,
the largest integer less than or equal to each element.

For integer inputs, follows the arra...

#### torch.floor_divide
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: floor_divide(input, other, *, out=None) -> Tensor

.. note::

    Before PyTorch 1.13 :func:`torch.floor_divide` incorrectly performed
    truncation division. To restore the previous behavior use
   ...

#### torch.frexp
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: frexp(input, *, out=None) -> (Tensor mantissa, Tensor exponent)

Decomposes :attr:`input` into mantissa and exponent tensors
such that :math:`\text{input} = \text{mantissa} \times 2^{\text{exponent}}`...

#### torch.isin
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: isin(elements, test_elements, *, assume_unique=False, invert=False) -> Tensor

Tests if each element of :attr:`elements` is in :attr:`test_elements`. Returns
a boolean tensor of the same shape as :att...

#### torch.isinf
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: isinf(input) -> Tensor

Tests if each element of :attr:`input` is infinite
(positive or negative infinity) or not.

.. note::
    Complex values are infinite when their real or imaginary part is
    i...

#### torch.isposinf
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: isposinf(input, *, out=None) -> Tensor
Tests if each element of :attr:`input` is positive infinity or not.

Args:
  input (Tensor): the input tensor.

Keyword args:
  out (Tensor, optional): the outpu...

#### torch.ldexp
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ldexp(input, other, *, out=None) -> Tensor

Multiplies :attr:`input` by 2 ** :attr:`other`.

.. math::
    \text{{out}}_i = \text{{input}}_i * 2^\text{{other}}_i


Typically this function is used to c...

#### torch.log
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: log(input, *, out=None) -> Tensor

Returns a new tensor with the natural logarithm of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{e} (x_{i})


Args:
    input (Tensor): the input tenso...

#### torch.log10
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: log10(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the logarithm to the base 10 of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{10} (x_{i})


Args:
    ...

#### torch.log1p
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: log1p(input, *, out=None) -> Tensor

Returns a new tensor with the natural logarithm of (1 + :attr:`input`).

.. math::
    y_i = \log_{e} (x_i + 1)

.. note:: This function is more accurate than :fun...

#### torch.log2
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: log2(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the logarithm to the base 2 of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{2} (x_{i})


Args:
    inp...

#### torch.logaddexp
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logaddexp(input, other, *, out=None) -> Tensor

Logarithm of the sum of exponentiations of the inputs.

Calculates pointwise :math:`\log\left(e^x + e^y\right)`. This function is useful
in statistics w...

#### torch.logaddexp2
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logaddexp2(input, other, *, out=None) -> Tensor

Logarithm of the sum of exponentiations of the inputs in base-2.

Calculates pointwise :math:`\log_2\left(2^x + 2^y\right)`. See
:func:`torch.logaddexp...

#### torch.logcumsumexp
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logcumsumexp(input, dim, *, out=None) -> Tensor
Returns the logarithm of the cumulative summation of the exponentiation of
elements of :attr:`input` in the dimension :attr:`dim`.

For summation index ...

#### torch.logdet
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logdet(input) -> Tensor

Calculates log determinant of a square matrix or batches of square matrices.

It returns ``-inf`` if the input has a determinant of zero, and ``NaN`` if it has
a negative dete...

#### torch.logical_and
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logical_and(input, other, *, out=None) -> Tensor

Computes the element-wise logical AND of the given input tensors. Zeros are treated as ``False`` and nonzeros are
treated as ``True``.

Args:
    inpu...

#### torch.logical_not
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logical_not(input, *, out=None) -> Tensor

Computes the element-wise logical NOT of the given input tensor. If not specified, the output tensor will have the bool
dtype. If the input tensor is not a b...

#### torch.logical_or
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logical_or(input, other, *, out=None) -> Tensor

Computes the element-wise logical OR of the given input tensors. Zeros are treated as ``False`` and nonzeros are
treated as ``True``.

Args:
    input ...

#### torch.logical_xor
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logical_xor(input: Tensor, other: Tensor, *, out: Optional[Tensor]) -> Tensor

Computes the element-wise logical XOR of the given input tensors. Zeros are treated as ``False`` and nonzeros are
treated...

#### torch.logit
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logit(input, eps=None, *, out=None) -> Tensor

Alias for :func:`torch.special.logit`....

#### torch.logspace
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logspace(start, end, steps, base=10.0, *,          out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor


Creates a one-dimensional tensor of size :attr:`steps` whos...

#### torch.logsumexp
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logsumexp(input, dim, keepdim=False, *, out=None)

Returns the log of summed exponentials of each row of the :attr:`input`
tensor in the given dimension :attr:`dim`. The computation is numerically
sta...

#### torch.matrix_exp
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: matrix_exp(A) -> Tensor

Alias for :func:`torch.linalg.matrix_exp`....

#### torch.pairwise_distance
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False) -> Tensor

See :class:`torch.nn.PairwiseDistance` for details...

#### torch.rsqrt
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: rsqrt(input, *, out=None) -> Tensor

Returns a new tensor with the reciprocal of the square-root of each of
the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{...

#### torch.sin
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sin(input, *, out=None) -> Tensor

Returns a new tensor with the sine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sin(\text{input}_{i})

Args:
    input (Tensor): the input tens...

#### torch.sinc
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sinc(input, *, out=None) -> Tensor

Alias for :func:`torch.special.sinc`....

#### torch.sinh
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sinh(input, *, out=None) -> Tensor

Returns a new tensor with the hyperbolic sine of the elements of
:attr:`input`.

.. math::
    \text{out}_{i} = \sinh(\text{input}_{i})

Args:
    input (Tensor): t...

#### torch.slogdet
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: slogdet(input) -> (Tensor, Tensor)

Alias for :func:`torch.linalg.slogdet`...

#### torch.sqrt
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sqrt(input, *, out=None) -> Tensor

Returns a new tensor with the square-root of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sqrt{\text{input}_{i}}

Args:
    input (Tensor): the i...

#### torch.tan
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: tan(input, *, out=None) -> Tensor

Returns a new tensor with the tangent of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \tan(\text{input}_{i})

Args:
    input (Tensor): the input t...

#### torch.tanh
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: tanh(input, *, out=None) -> Tensor

Returns a new tensor with the hyperbolic tangent of the elements
of :attr:`input`.

.. math::
    \text{out}_{i} = \tanh(\text{input}_{i})

Args:
    input (Tensor)...

#### torch.xlogy
- **Category**: TORCH_MATH_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: xlogy(input, other, *, out=None) -> Tensor

Alias for :func:`torch.special.xlogy`....

#### torch.adaptive_avg_pool1d
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: adaptive_avg_pool1d(input, output_size) -> Tensor

Applies a 1D adaptive average pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveAvgPool1d` for details an...

#### torch.avg_pool1d
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor

Applies a 1D average pooling over an input signal composed of several
input planes.

See :cla...

#### torch.bilinear
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bilinear(input1, input2, weight, bias=None) -> Tensor

Applies a bilinear transformation to the incoming data:
:math:`y = x_1^T A x_2 + b`

Shape:

    - input1: :math:`(N, *, H_{in1})` where :math:`H...

#### torch.conv1d
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

This operator supports :ref:`Ten...

#### torch.conv2d
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 2D convolution over an input image composed of several input
planes.

This operator supports :ref:`Tens...

#### torch.conv3d
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 3D convolution over an input image composed of several input
planes.

This operator supports :ref:`Tens...

#### torch.conv_tbc
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Applies a 1-dimensional sequence convolution over an input sequence.
Input and output dimensions are (Time, Batch, Channels) - hence TBC.

Args:
    input: input tensor of shape :math:`(\text{sequence...

#### torch.conv_transpose1d
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 1D transposed convolution operator over an input signal
composed of several...

#### torch.conv_transpose2d
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 2D transposed convolution operator over an input image
composed of several ...

#### torch.conv_transpose3d
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 3D transposed convolution operator over an input image
composed of several ...

#### torch.prelu
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: prelu(input, weight) -> Tensor

Applies element-wise the function
:math:`\text{PReLU}(x) = \max(0,x) + \text{weight} * \min(0,x)` where weight is a
learnable parameter.

.. note::
    `weight` is expe...

#### torch.quantized_max_pool1d
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: quantized_max_pool1d(input, kernel_size, stride=[], padding=0, dilation=1, ceil_mode=False) -> Tensor

Applies a 1D max pooling over an input quantized tensor composed of several input planes.

Argume...

#### torch.quantized_max_pool2d
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: quantized_max_pool2d(input, kernel_size, stride=[], padding=0, dilation=1, ceil_mode=False) -> Tensor

Applies a 2D max pooling over an input quantized tensor composed of several input planes.

Argume...

#### torch.relu_
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: relu_(input) -> Tensor

In-place version of :func:`~relu`....

#### torch.rrelu_
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: rrelu_(input, lower=1./8, upper=1./3, training=False) -> Tensor

In-place version of :func:`~rrelu`....

#### torch.sigmoid
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sigmoid(input, *, out=None) -> Tensor

Alias for :func:`torch.special.expit`....

#### torch.softmax
- **Category**: TORCH_NN_OPS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: softmax(input, dim, *, dtype=None) -> Tensor

Alias for :func:`torch.nn.functional.softmax`....

#### torch.bernoulli
- **Category**: TORCH_RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bernoulli(input: Tensor, *, generator: Optional[Generator], out: Optional[Tensor]) -> Tensor

Draws binary random numbers (0 or 1) from a Bernoulli distribution.

The :attr:`input` tensor should be a ...

#### torch.normal
- **Category**: TORCH_RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: normal(mean, std, *, generator=None, out=None) -> Tensor

Returns a tensor of random numbers drawn from separate normal distributions
whose mean and standard deviation are given.

The :attr:`mean` is ...

#### torch.rand
- **Category**: TORCH_RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: rand(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor

Returns a tensor filled with random numbers from a uniform dis...

#### torch.rand_like
- **Category**: TORCH_RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: rand_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor with the same size as :attr:`input` that is filled with
...

#### torch.randint
- **Category**: TORCH_RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: randint(low=0, high, size, \*, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with random integers generated uniformly...

#### torch.randint_like
- **Category**: TORCH_RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: randint_like(input, low=0, high, \*, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor with the same shape as Tensor ...

#### torch.randn
- **Category**: TORCH_RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: randn(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor


Returns a tensor filled with random numbers from a normal di...

#### torch.randn_like
- **Category**: TORCH_RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor with the same size as :attr:`input` that is filled with...

#### torch.randperm
- **Category**: TORCH_RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: randperm(n, *, generator=None, out=None, dtype=torch.int64,layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor

Returns a random permutation of integers from ``0`` to `...

#### torch.set_flush_denormal
- **Category**: TORCH_RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: set_flush_denormal(mode) -> bool

Disables denormal floating numbers on CPU.

Returns ``True`` if your system supports flushing denormal numbers and it
successfully configures flush denormal mode.  :m...

#### torch._dynamo.export
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Export an input function f to a format that can be executed outside of PyTorch using the FX graph.

Args:
    f (callable): A PyTorch function to be exported.

    aten_graph (bool): If True, exports ...

#### torch._dynamo.mark_dynamic
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Mark a tensor as having a dynamic dim and set corresponding min and max range for the dim.

[Note - on the state of mark_dynamic]

The behavior of having a dynamic dimension on a tensor is governed by...

#### torch._dynamo.mark_static
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Mark a tensor as having a static dim or mark a nn module class as static.

For tensors
===========
This will prevent us from attempting to compile it dynamically
when dynamic=True; this can improve tr...

#### torch._dynamo.mark_static_address
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Marks an input tensor whose data_ptr will not change across multiple calls
to a dynamo-compiled function. This indicates to cudagraphs that an extra allocation
is not needed for this input. The data_p...

#### torch._dynamo.maybe_mark_dynamic
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Mark a tensor as having a dynamic dim, but don't enforce it (i.e., if this
dimension ends up getting specialized, don't error)....

#### torch._dynamo.register_backend
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Decorator to add a given compiler to the registry to allow calling
`torch.compile` with string shorthand.  Note: for projects not
imported by default, it might be easier to pass a function directly
as...

#### torch._export.ExportGraphSignature
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: :class:`ExportGraphSignature` models the input/output signature of Export Graph,
which is a fx.Graph with stronger invariants gurantees.

Export Graph is functional and does not access "states" like p...

#### torch._export.InputSpec
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: InputSpec(kind: torch.export.graph_signature.InputKind, arg: Union[torch.export.graph_signature.TensorArgument, torch.export.graph_signature.SymIntArgument, torch.export.graph_signature.SymFloatArgume...

#### torch._export.OutputSpec
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: OutputSpec(kind: torch.export.graph_signature.OutputKind, arg: Union[torch.export.graph_signature.TensorArgument, torch.export.graph_signature.SymIntArgument, torch.export.graph_signature.SymFloatArgu...

#### torch._export.aot_compile
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Note: this function is not stable yet

Traces either an nn.Module's forward function or just a callable with PyTorch
operations inside, generates executable cpp code from the program, and returns
the ...

#### torch._higher_order_ops.associative_scan
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs an inclusive scan with an associative combine function.

.. warning::
    `torch.associative_scan` is a prototype feature in PyTorch. It currently
    does not support autograd and you may ru...

#### torch._higher_order_ops.cond
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Conditionally applies `true_fn` or `false_fn`.

.. warning::
    `torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types and
    doesn't support training curr...

#### torch._higher_order_ops.scan
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs an inclusive scan with a combine function.

.. warning::
    `torch.scan` is a prototype feature in PyTorch. It currently
    does not support autograd and you may run into miscompiles.
    R...

#### torch._higher_order_ops.while_loop
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Run body_fn(*carried_inputs) while cond_fn(*carried_inputs) returns a True scalar tensor. Returns the output of body_fn or
initial carried_inputs.

.. warning::
    `torch.while_loop` is a prototype f...

#### torch._inductor.aoti_load_package
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Loads the model from the PT2 package.

If multiple models were packaged into the PT2, this will load the default
model. To load a specific model, you can directly call the load API

.. code-block:: py...

#### torch._inductor.compile
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Compile a given FX graph with TorchInductor.  This allows compiling
FX graphs captured without using TorchDynamo.

Args:
    gm: The FX graph to compile.
    example_inputs:  List of tensor inputs.
  ...

#### torch._lazy.add_step_closure
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Adds a closure to the list of the ones to be run at the end of the step.
Many times during model training there is the need to print/report (print to
console, post to tensorboard, etc...) information ...

#### torch._lazy.mark_step
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Triggers a mark step, which amounts to
- collecting a group of 'live' lazy tensors to index into the compilation cache
  (lowering/compiling their IR graphs if not cached)
- kicking off execution of t...

#### torch._lazy.sync_multi
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sync the list of lazy tensors so there IR get lowered for the activate backend
and the compiled computation graph get cached....

#### torch._lazy.to_cpu
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch._library.register_fake_class
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Register a fake implementation for this class.

It's in the same spirit of registering a fake implementation for
an operator but with the difference that it
associates a fake class with the original t...

#### torch._library.triton_op
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Create a custom operator whose implementation is backed by 1+ triton kernels.

This is a more structured way of using triton kernels with PyTorch.
Prefer using triton kernels with no ``torch.library``...

#### torch._library.wrap_triton
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Allows capture of a triton kernel into a graph via make_fx or
non-strict ``torch.export``.

These technologies perform Dispatcher-based tracing (via
``__torch_dispatch__``) and cannot see calls to raw...

#### torch._logging.set_logs
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the log level for individual components and toggles individual log
artifact types.

.. warning:: This feature is a prototype and may have compatibility
    breaking changes in the future.

.. not...

#### torch._numpy.broadcast_shapes
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: broadcast_shapes(*shapes) -> Size

Similar to :func:`broadcast_tensors` but for shapes.

This is equivalent to
``torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape``
but avoids the need creat...

#### torch._prims.has_torch_function
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Check for __torch_function__ implementations in the elements of an iterable
or if a __torch_function__ mode is enabled.  Considers exact ``Tensor`` s
and ``Parameter`` s non-dispatchable.  Use this to...

#### torch._prims_common.check_same_dtype
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Checks that all Tensors in args have the same device and that all Numbers have the
same corresponding Python type.

Raises a RuntimeError when:
  - args contains an object whose type is not Tensor or ...

#### torch._prims_common.check_same_shape
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Checks that all Tensors in args have the same shape.

Raises a RuntimeError when:
  - args contains an object whose type is not Tensor or Number
  - two Tensor objects in args have different devices...

#### torch._prims_common.compute_required_storage_length
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Computes the minimum storage size to hold the given tensor geometry.

Example
=======

This is the size of a newly allocated tensor's storage, in units of elements

>>> t = torch.empty((10, 20))
>>> c...

#### torch._prims_common.elementwise_dtypes
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Computes the computation and result dtypes for elementwise type promotion
on the given arguments and with the given elementwise type promotion kind.

Note that not all inputs to an elementwise operati...

#### torch._prims_common.is_contiguous
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Tests whether a tensor is contiguous or not.

Tensors are contiguous when they have no elements,
one element, or when they have "nested" strides....

#### torch._prims_common.is_non_overlapping_and_dense
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: True when a tensor is non-overlapping and dense.

A tensor is non-overlapping and dense when there exists a permutation of
its dimensions that is contiguous....

#### torch._prims_common.make_contiguous_strides_for
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the strides of a contiguous tensor if row_major
If row_major=True, it returns the strides of a contiguous batch of Fortran-contiguous matrices
This is often used when calling external librarie...

#### torch._refs.DispatchKey
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Members:

Undefined

CompositeExplicitAutogradNonFunctional

CompositeExplicitAutograd

CompositeImplicitAutogradNestedTensor

CompositeImplicitAutograd

AutogradNestedTensor

AutogradOther

Autograd
...

#### torch._refs.block_diag
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: This is used as an input to PythonRefInfo. `torch.block_diag`
expects arguments splatted, but `aten.block_diag` expects only
one argument that is a list of Tensors....

#### torch._refs.elementwise_unary_scalar_wrapper
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Allows unary operators that accept tensors to work with Python numbers....

#### torch._refs.new_empty_strided
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Reference implementation of torch.Tensor.new_empty_strided...

#### torch._subclasses.CrossRefFakeMode
- **Category**: CORE_TORCH
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: A ``TorchDispatchMode`` allows you to override the meaning of all
``__torch_dispatch__`` overrideable functions within a dynamic scope,
without having to actually create a tensor subclass or manually
...

#### torch.accelerator.current_accelerator
- **Category**: ACCELERATOR_SUPPORT
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return the device of the accelerator available at compilation time.
If no accelerator were available at compilation time, returns None.
See :ref:`accelerator<accelerators>` for details.

Args:
    che...

#### torch.accelerator.is_available
- **Category**: ACCELERATOR_SUPPORT
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Check if the current accelerator is available at runtime: it was build, all the
required drivers are available and at least one device is visible.
See :ref:`accelerator<accelerators>` for details.

Re...

#### torch.accelerator.synchronize
- **Category**: ACCELERATOR_SUPPORT
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Wait for all kernels in all streams on the given device to complete.

Args:
    device (:class:`torch.device`, str, int, optional): device for which to synchronize. It must match
        the current :...

#### torch.amp.GradScaler
- **Category**: AUTOMATIC_MIXED_PRECISION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: An instance ``scaler`` of :class:`GradScaler`.

Helps perform the steps of gradient scaling
conveniently.

* ``scaler.scale(loss)`` multiplies a given loss by ``scaler``'s current scale factor.
* ``sc...

#### torch.amp.autocast
- **Category**: AUTOMATIC_MIXED_PRECISION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Instances of :class:`autocast` serve as context managers or decorators that
allow regions of your script to run in mixed precision.

In these regions, ops run in an op-specific dtype chosen by autocas...

#### torch.amp.custom_bwd
- **Category**: AUTOMATIC_MIXED_PRECISION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Create a helper decorator for backward methods of custom autograd functions.

Autograd functions are subclasses of :class:`torch.autograd.Function`.
Ensures that ``backward`` executes with the same au...

#### torch.amp.custom_fwd
- **Category**: AUTOMATIC_MIXED_PRECISION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Create a helper decorator for ``forward`` methods of custom autograd functions.

Autograd functions are subclasses of :class:`torch.autograd.Function`.
See the :ref:`example page<amp-custom-examples>`...

#### torch.amp.is_autocast_available
- **Category**: AUTOMATIC_MIXED_PRECISION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return a bool indicating if autocast is available on :attr:`device_type`.

Args:
    device_type(str):  Device type to use. Possible values are: 'cuda', 'cpu', 'mtia', 'maia', 'xpu', and so on.
      ...

#### torch.autograd.Function
- **Category**: AUTOGRAD
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Base class to create custom `autograd.Function`.

To create a custom `autograd.Function`, subclass this class and implement
the :meth:`forward` and :meth:`backward` static methods. Then, to use your c...

#### torch.autograd.ProfilerActivity
- **Category**: AUTOGRAD
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Members:

CPU

XPU

MTIA

CUDA

HPU

PrivateUse1...

#### torch.autograd.ProfilerState
- **Category**: AUTOGRAD
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Members:

Disabled

CPU

CUDA

NVTX

ITT

PRIVATEUSE1

KINETO

KINETO_GPU_FALLBACK

KINETO_PRIVATEUSE1_FALLBACK...

#### torch.autograd.backward
- **Category**: AUTOGRAD
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Compute the sum of gradients of given tensors with respect to graph leaves.

The graph is differentiated using the chain rule. If any of ``tensors``
are non-scalar (i.e. their data has more than one e...

#### torch.autograd.detect_anomaly
- **Category**: AUTOGRAD
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Context-manager that enable anomaly detection for the autograd engine.

This does two things:

- Running the forward pass with detection enabled will allow the backward
  pass to print the traceback o...

#### torch.autograd.enable_grad
- **Category**: AUTOGRAD
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Context-manager that enables gradient calculation.

Enables gradient calculation, if it has been disabled via :class:`~no_grad`
or :class:`~set_grad_enabled`.

This context manager is thread local; it...

#### torch.autograd.grad
- **Category**: AUTOGRAD
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Compute and return the sum of gradients of outputs with respect to the inputs.

``grad_outputs`` should be a sequence of length matching ``output``
containing the "vector" in vector-Jacobian product, ...

#### torch.autograd.gradcheck
- **Category**: AUTOGRAD
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Check gradients computed via small finite differences against analytical
gradients wrt tensors in :attr:`inputs` that are of floating point or complex type
and with ``requires_grad=True``.

The check ...

#### torch.autograd.gradgradcheck
- **Category**: AUTOGRAD
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Check gradients of gradients computed via small finite differences
against analytical gradients wrt tensors in :attr:`inputs` and
:attr:`grad_outputs` that are of floating point or complex type and wi...

#### torch.autograd.has_torch_function
- **Category**: AUTOGRAD
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Check for __torch_function__ implementations in the elements of an iterable
or if a __torch_function__ mode is enabled.  Considers exact ``Tensor`` s
and ``Parameter`` s non-dispatchable.  Use this to...

#### torch.autograd.inference_mode
- **Category**: AUTOGRAD
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Context-manager that enables or disables inference mode.

InferenceMode is a context manager analogous to :class:`~no_grad`
to be used when you are certain your operations will have no interactions
wi...

#### torch.autograd.no_grad
- **Category**: AUTOGRAD
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Context-manager that disables gradient calculation.

Disabling gradient calculation is useful for inference, when you are sure
that you will not call :meth:`Tensor.backward()`. It will reduce memory
c...

#### torch.autograd.set_grad_enabled
- **Category**: AUTOGRAD
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Context-manager that sets gradient calculation on or off.

``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
It can be used as a context-manager or as a function.

...

#### torch.compiler.allow_in_graph
- **Category**: COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Tells the compiler frontend (Dynamo) to skip symbolic introspection of the function
and instead directly write it to the graph when encountered.

If you are using :func:`torch.compile` (with backend="...

#### torch.compiler.wrap_numpy
- **Category**: COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Decorator that turns a function from ``np.ndarray``s to ``np.ndarray``s into a function
from ``torch.Tensor``s to ``torch.Tensor``s.

It is designed to be used with :func:`torch.compile` with ``fullgr...

#### torch.cpu.AbstractContextManager
- **Category**: CPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: An abstract base class for context managers....

#### torch.cpu.Any
- **Category**: CPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements...

#### torch.cpu.is_available
- **Category**: CPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns a bool indicating if CPU is currently available.

N.B. This function only exists to facilitate device-agnostic code...

#### torch.cpu.synchronize
- **Category**: CPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Waits for all kernels in all streams on the CPU device to complete.

Args:
    device (torch.device or int, optional): ignored, there's only one CPU device.

N.B. This function only exists to facilita...

#### torch.distributed.GradBucket
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: This class mainly passes a flattened gradient tensor
(returned by :meth:`~torch.distributed.GradBucket.buffer`)
to DDP communication hook.
This tensor can be further decomposed into a list of per-para...

#### torch.distributed.P2POp
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: A class to build point-to-point operations for ``batch_isend_irecv``.

This class builds the type of P2P operation, communication buffer, peer rank,
Process Group, and tag. Instances of this class wil...

#### torch.distributed.ReduceOp
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: An enum-like class for available reduction operations: ``SUM``, ``PRODUCT``,
``MIN``, ``MAX``, ``BAND``, ``BOR``, ``BXOR``, and ``PREMUL_SUM``.

``BAND``, ``BOR``, and ``BXOR`` reductions are not avai...

#### torch.distributed.Work
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: A `Work` object represents the handle to a pending asynchronous operation in
PyTorch's distributed package. It is returned by non-blocking collective operations,
such as `dist.all_reduce(tensor, async...

#### torch.distributed.all_gather
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Gathers tensors from the whole group in a list.

Complex and uneven sized tensors are supported.

Args:
    tensor_list (list[Tensor]): Output list. It should contain
        correctly-sized tensors t...

#### torch.distributed.all_gather_coalesced
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Gathers input tensors from the whole group in a list in a coalesced manner.

Complex tensors are supported.

Args:
    output_tensor_lists (list[list[Tensor]]): Output list. It should contain
        ...

#### torch.distributed.all_gather_object
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Gathers picklable objects from the whole group into a list.

Similar to :func:`all_gather`, but Python objects can be passed in.
Note that the object must be picklable in order to be gathered.

Args:
...

#### torch.distributed.all_reduce
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Reduces the tensor data across all machines in a way that all get the final result.

After the call ``tensor`` is going to be bitwise identical in all processes.

Complex tensors are supported.

Args:...

#### torch.distributed.all_reduce_coalesced
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: WARNING: at this time individual shape checking is not implemented across nodes.

For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the
rank 1 node passes [torch.rand(2), torch...

#### torch.distributed.all_to_all
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.

Complex tensors are supported.

Args:
    output_tensor_list (list[Tensor]): List of ten...

#### torch.distributed.all_to_all_single
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Split input tensor and then scatter the split list to all processes in a group.

Later the received tensors are concatenated from all the processes in the group
and returned as a single output tensor....

#### torch.distributed.barrier
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Synchronize all processes.

This collective blocks processes until the whole group enters this function,
if async_op is False, or if async work handle is called on wait().

Args:
    group (ProcessGro...

#### torch.distributed.batch_isend_irecv
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Send or Receive a batch of tensors asynchronously and return a list of requests.

Process each of the operations in ``p2p_op_list`` and return the corresponding
requests. NCCL, Gloo, and UCC backend a...

#### torch.distributed.broadcast
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Broadcasts the tensor to the whole group.

``tensor`` must have the same number of elements in all processes
participating in the collective.

Args:
    tensor (Tensor): Data to be sent if ``src`` is ...

#### torch.distributed.broadcast_object_list
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Broadcasts picklable objects in ``object_list`` to the whole group.

Similar to :func:`broadcast`, but Python objects can be passed in.
Note that all objects in ``object_list`` must be picklable in or...

#### torch.distributed.gather
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Gathers a list of tensors in a single process.

This function requires all tensors to be the same size on each process.

Args:
    tensor (Tensor): Input tensor.
    gather_list (list[Tensor], optiona...

#### torch.distributed.gather_object
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Gathers picklable objects from the whole group in a single process.

Similar to :func:`gather`, but Python objects can be passed in. Note that the
object must be picklable in order to be gathered.

Ar...

#### torch.distributed.get_node_local_rank
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return the local rank of the current process relative to the node.

Semantically, this is a useful concept for mapping processes to devices.
For example, on a node with 8 accelerator you could use the...

#### torch.distributed.init_process_group
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Initialize the default distributed process group.

This will also initialize the distributed package.

There are 2 main ways to initialize a process group:
    1. Specify ``store``, ``rank``, and ``wo...

#### torch.distributed.irecv
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Receives a tensor asynchronously.

.. warning::
    ``tag`` is not supported with the NCCL backend.

Unlike recv, which is blocking, irecv allows src == dst rank, i.e. recv from self.

Args:
    tenso...

#### torch.distributed.isend
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Send a tensor asynchronously.

.. warning::
    Modifying ``tensor`` before the request completes causes undefined
    behavior.

.. warning::
    ``tag`` is not supported with the NCCL backend.

Unli...

#### torch.distributed.new_group
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Create a new distributed group.

This function requires that all processes in the main group (i.e. all
processes that are part of the distributed job) enter this function, even
if they are not going t...

#### torch.distributed.new_subgroups
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Create subgroups of equal size.

By default, it creates intra-machine subgroups,
where each of which contains all the ranks of a machine, based on the assumption
that each machine has the same number ...

#### torch.distributed.new_subgroups_by_enumeration
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Create subgroups by dividing the global world.

The division is specified by a nested list of ranks. The subgroups cannot have
overlap, and some ranks may not have to be in any subgroup.

This is a co...

#### torch.distributed.recv
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Receives a tensor synchronously.

.. warning::
    ``tag`` is not supported with the NCCL backend.

Args:
    tensor (Tensor): Tensor to fill with received data.
    src (int, optional): Source rank o...

#### torch.distributed.recv_object_list
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Receives picklable objects in ``object_list`` synchronously.

Similar to :func:`recv`, but can receive Python objects.

Args:
    object_list (List[Any]): List of objects to receive into.
        Must...

#### torch.distributed.reduce
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Reduces the tensor data across all machines.

Only the process with rank ``dst`` is going to receive the final result.

Args:
    tensor (Tensor): Input and output of the collective. The function
    ...

#### torch.distributed.reduce_scatter
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Reduces, then scatters a list of tensors to all processes in a group.

Args:
    output (Tensor): Output tensor.
    input_list (list[Tensor]): List of tensors to reduce and scatter.
    op (optional)...

#### torch.distributed.scatter
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Scatters a list of tensors to all processes in a group.

Each process will receive exactly one tensor and store its data in the
``tensor`` argument.

Complex tensors are supported.

Args:
    tensor (...

#### torch.distributed.scatter_object_list
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Scatters picklable objects in ``scatter_object_input_list`` to the whole group.

Similar to :func:`scatter`, but Python objects can be passed in. On
each rank, the scattered object will be stored as t...

#### torch.distributed.send
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Send a tensor synchronously.

.. warning::
    ``tag`` is not supported with the NCCL backend.

Args:
    tensor (Tensor): Tensor to send.
    dst (int): Destination rank on global process group (rega...

#### torch.distributed.send_object_list
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sends picklable objects in ``object_list`` synchronously.

Similar to :func:`send`, but Python objects can be passed in.
Note that all objects in ``object_list`` must be picklable in order to be
sent....

#### torch.distributed.split_group
- **Category**: DISTRIBUTED_COMPUTING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Create a new process group splitted from the given parent process group.

warning:: This is an experimental API and only the ``NCCL`` backend supports this API.
Other backends will raise an error.
Use...

#### torch.distributions.AffineTransform
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Transform via the pointwise affine mapping :math:`y = \text{loc} + \text{scale} \times x`.

Args:
    loc (Tensor or float): Location parameter.
    scale (Tensor or float): Scale parameter.
    event...

#### torch.distributions.Bernoulli
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Bernoulli distribution parameterized by :attr:`probs`
or :attr:`logits` (but not both).

Samples are binary (0 or 1). They take the value `1` with probability `p`
and `0` with probability `1...

#### torch.distributions.Beta
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`.

Example::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m = Beta(torch.tensor([0.5]), torch.tens...

#### torch.distributions.Binomial
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Binomial distribution parameterized by :attr:`total_count` and
either :attr:`probs` or :attr:`logits` (but not both). :attr:`total_count` must be
broadcastable with :attr:`probs`/:attr:`logi...

#### torch.distributions.Categorical
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a categorical distribution parameterized by either :attr:`probs` or
:attr:`logits` (but not both).

.. note::
    It is equivalent to the distribution that :func:`torch.multinomial`
    sample...

#### torch.distributions.Cauchy
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Samples from a Cauchy (Lorentz) distribution. The distribution of the ratio of
independent normally distributed random variables with means `0` follows a
Cauchy distribution.

Example::

    >>> # xdo...

#### torch.distributions.Chi2
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Chi-squared distribution parameterized by shape parameter :attr:`df`.
This is exactly equivalent to ``Gamma(alpha=0.5*df, beta=0.5)``

Example::

    >>> # xdoctest: +IGNORE_WANT("non-determ...

#### torch.distributions.ContinuousBernoulli
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a continuous Bernoulli distribution parameterized by :attr:`probs`
or :attr:`logits` (but not both).

The distribution is supported in [0, 1] and parameterized by 'probs' (in
(0,1)) or 'logits...

#### torch.distributions.Dirichlet
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Dirichlet distribution parameterized by concentration :attr:`concentration`.

Example::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m = Dirichlet(torch.tensor([0.5, 0.5]))...

#### torch.distributions.Exponential
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Exponential distribution parameterized by :attr:`rate`.

Example::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m = Exponential(torch.tensor([1.0]))
    >>> m.sample()  # E...

#### torch.distributions.FisherSnedecor
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Fisher-Snedecor distribution parameterized by :attr:`df1` and :attr:`df2`.

Example::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m = FisherSnedecor(torch.tensor([1.0]), t...

#### torch.distributions.Gamma
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`.

Example::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m = Gamma(torch.tensor([1.0]), tor...

#### torch.distributions.GeneralizedPareto
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Generalized Pareto distribution parameterized by :attr:`loc`, :attr:`scale`, and :attr:`concentration`.

The Generalized Pareto distribution is a family of continuous probability distributio...

#### torch.distributions.Geometric
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Geometric distribution parameterized by :attr:`probs`,
where :attr:`probs` is the probability of success of Bernoulli trials.

.. math::

    P(X=k) = (1-p)^{k} p, k = 0, 1, ...

.. note::
 ...

#### torch.distributions.Gumbel
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Samples from a Gumbel Distribution.

Examples::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))
    >>> m.sample()  # sample from Gu...

#### torch.distributions.HalfCauchy
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a half-Cauchy distribution parameterized by `scale` where::

    X ~ Cauchy(0, scale)
    Y = |X| ~ HalfCauchy(scale)

Example::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> ...

#### torch.distributions.HalfNormal
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a half-normal distribution parameterized by `scale` where::

    X ~ Normal(0, scale)
    Y = |X| ~ HalfNormal(scale)

Example::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> ...

#### torch.distributions.InverseGamma
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates an inverse gamma distribution parameterized by :attr:`concentration` and :attr:`rate`
where::

    X ~ Gamma(concentration, rate)
    Y = 1 / X ~ InverseGamma(concentration, rate)

Example::

...

#### torch.distributions.Kumaraswamy
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Samples from a Kumaraswamy distribution.

Example::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m = Kumaraswamy(torch.tensor([1.0]), torch.tensor([1.0]))
    >>> m.sample()  # sampl...

#### torch.distributions.LKJCholesky
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: LKJ distribution for lower Cholesky factor of correlation matrices.
The distribution is controlled by ``concentration`` parameter :math:`\eta`
to make the probability of the correlation matrix :math:`...

#### torch.distributions.Laplace
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.

Example::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m = Laplace(torch.tensor([0.0]), torch.tensor([...

#### torch.distributions.LogNormal
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a log-normal distribution parameterized by
:attr:`loc` and :attr:`scale` where::

    X ~ Normal(loc, scale)
    Y = exp(X) ~ LogNormal(loc, scale)

Example::

    >>> # xdoctest: +IGNORE_WANT...

#### torch.distributions.LogisticNormal
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a logistic-normal distribution parameterized by :attr:`loc` and :attr:`scale`
that define the base `Normal` distribution transformed with the
`StickBreakingTransform` such that::

    X ~ Logi...

#### torch.distributions.LowRankMultivariateNormal
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a multivariate normal distribution with covariance matrix having a low-rank form
parameterized by :attr:`cov_factor` and :attr:`cov_diag`::

    covariance_matrix = cov_factor @ cov_factor.T +...

#### torch.distributions.Multinomial
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Multinomial distribution parameterized by :attr:`total_count` and
either :attr:`probs` or :attr:`logits` (but not both). The innermost dimension of
:attr:`probs` indexes over categories. All...

#### torch.distributions.MultivariateNormal
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a multivariate normal (also called Gaussian) distribution
parameterized by a mean vector and a covariance matrix.

The multivariate normal distribution can be parameterized either
in terms of ...

#### torch.distributions.NegativeBinomial
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Negative Binomial distribution, i.e. distribution
of the number of successful independent and identical Bernoulli trials
before :attr:`total_count` failures are achieved. The probability
of ...

#### torch.distributions.Normal
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a normal (also called Gaussian) distribution parameterized by
:attr:`loc` and :attr:`scale`.

Example::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m = Normal(torch.tensor([...

#### torch.distributions.OneHotCategorical
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a one-hot categorical distribution parameterized by :attr:`probs` or
:attr:`logits`.

Samples are one-hot coded vectors of size ``probs.size(-1)``.

.. note:: The `probs` argument must be non-...

#### torch.distributions.Pareto
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Samples from a Pareto Type 1 distribution.

Example::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
    >>> m.sample()  # sample f...

#### torch.distributions.Poisson
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Poisson distribution parameterized by :attr:`rate`, the rate parameter.

Samples are nonnegative integers, with a pmf given by

.. math::
  \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!}

Exa...

#### torch.distributions.RelaxedBernoulli
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a RelaxedBernoulli distribution, parametrized by
:attr:`temperature`, and either :attr:`probs` or :attr:`logits`
(but not both). This is a relaxed version of the `Bernoulli` distribution,
so t...

#### torch.distributions.RelaxedOneHotCategorical
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a RelaxedOneHotCategorical distribution parametrized by
:attr:`temperature`, and either :attr:`probs` or :attr:`logits`.
This is a relaxed version of the :class:`OneHotCategorical` distributio...

#### torch.distributions.ReshapeTransform
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Unit Jacobian transform to reshape the rightmost part of a tensor.

Note that ``in_shape`` and ``out_shape`` must have the same number of
elements, just as for :meth:`torch.Tensor.reshape`.

Arguments...

#### torch.distributions.StudentT
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Student's t-distribution parameterized by degree of
freedom :attr:`df`, mean :attr:`loc` and scale :attr:`scale`.

Example::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m ...

#### torch.distributions.Transform
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Abstract class for invertable transformations with computable log
det jacobians. They are primarily used in
:class:`torch.distributions.TransformedDistribution`.

Caching is useful for transforms whos...

#### torch.distributions.Uniform
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Generates uniformly distributed random samples from the half-open interval
``[low, high)``.

Example::

    >>> m = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))
    >>> m.sample()  # uniformly di...

#### torch.distributions.VonMises
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: A circular von Mises distribution.

This implementation uses polar coordinates. The ``loc`` and ``value`` args
can be any real number (to facilitate unconstrained optimization), but are
interpreted as...

#### torch.distributions.Weibull
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Samples from a two-parameter Weibull distribution.

Example:

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> m = Weibull(torch.tensor([1.0]), torch.tensor([1.0]))
    >>> m.sample()  # ...

#### torch.distributions.Wishart
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Creates a Wishart distribution parameterized by a symmetric positive definite matrix :math:`\Sigma`,
or its Cholesky decomposition :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`

Example:
    >>>...

#### torch.distributions.kl_divergence
- **Category**: DISTRIBUTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions.

.. math::

    KL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)} \,dx

Args:
    p (Distribution): A :class:`~torch.dist...

#### torch.export.Dim
- **Category**: MODEL_EXPORT
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: :func:`Dim` constructs a type analogous to a named symbolic integer with a range.
It can be used to describe multiple possible values of a dynamic tensor dimension.
Note that different dynamic dimensi...

#### torch.export.ExportGraphSignature
- **Category**: MODEL_EXPORT
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: :class:`ExportGraphSignature` models the input/output signature of Export Graph,
which is a fx.Graph with stronger invariants gurantees.

Export Graph is functional and does not access "states" like p...

#### torch.export.ExportedProgram
- **Category**: MODEL_EXPORT
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Package of a program from :func:`export`. It contains
an :class:`torch.fx.Graph` that represents Tensor computation, a state_dict containing
tensor values of all lifted parameters and buffers, and var...

#### torch.export.ModuleCallSignature
- **Category**: MODEL_EXPORT
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ModuleCallSignature(inputs: list[typing.Union[torch.export.graph_signature.TensorArgument, torch.export.graph_signature.SymIntArgument, torch.export.graph_signature.SymFloatArgument, torch.export.grap...

#### torch.export.ShapesCollection
- **Category**: MODEL_EXPORT
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Builder for dynamic_shapes.
Used to assign dynamic shape specifications to tensors that appear in inputs.

This is useful particularly when :func:`args` is a nested input structure, and it's
easier to...

#### torch.export.export
- **Category**: MODEL_EXPORT
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: :func:`export` takes any nn.Module along with example inputs, and produces a traced graph representing
only the Tensor computation of the function in an Ahead-of-Time (AOT) fashion,
which can subseque...

#### torch.export.export_for_training
- **Category**: MODEL_EXPORT
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: :func:`export_for_training` takes any nn.Module along with example inputs, and produces a traced graph representing
only the Tensor computation of the function in an Ahead-of-Time (AOT) fashion,
which...

#### torch.export.register_dataclass
- **Category**: MODEL_EXPORT
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Registers a dataclass as a valid input/output type for :func:`torch.export.export`.

Args:
    cls: the dataclass type to register
    serialized_type_name: The serialized name for the dataclass. This...

#### torch.fft.fft
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: fft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the one dimensional discrete Fourier transform of :attr:`input`.

Note:
    The Fourier domain representation of any real signal ...

#### torch.fft.fft2
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: fft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the 2 dimensional discrete Fourier transform of :attr:`input`.
Equivalent to :func:`~torch.fft.fftn` but FFTs only the las...

#### torch.fft.fftfreq
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: fftfreq(n, d=1.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Computes the discrete Fourier Transform sample frequencies for a signal of size :attr:`n`....

#### torch.fft.fftn
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: fftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

Computes the N dimensional discrete Fourier transform of :attr:`input`.

Note:
    The Fourier domain representation of any real signal...

#### torch.fft.fftshift
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: fftshift(input, dim=None) -> Tensor

Reorders n-dimensional FFT data, as provided by :func:`~torch.fft.fftn`, to have
negative frequency terms first.

This performs a periodic shift of n-dimensional d...

#### torch.fft.hfft
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: hfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the one dimensional discrete Fourier transform of a Hermitian
symmetric :attr:`input` signal.

Note:

    :func:`~torch.fft.hfft...

#### torch.fft.hfft2
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: hfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the 2-dimensional discrete Fourier transform of a Hermitian symmetric
:attr:`input` signal. Equivalent to :func:`~torch.f...

#### torch.fft.hfftn
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: hfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

Computes the n-dimensional discrete Fourier transform of a Hermitian symmetric
:attr:`input` signal.

:attr:`input` is interpreted as ...

#### torch.fft.ifft
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ifft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the one dimensional inverse discrete Fourier transform of :attr:`input`.

Note:
    Supports torch.half and torch.chalf on CUDA ...

#### torch.fft.ifft2
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ifft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the 2 dimensional inverse discrete Fourier transform of :attr:`input`.
Equivalent to :func:`~torch.fft.ifftn` but IFFTs o...

#### torch.fft.ifftn
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ifftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

Computes the N dimensional inverse discrete Fourier transform of :attr:`input`.

Note:
    Supports torch.half and torch.chalf on CUDA...

#### torch.fft.ifftshift
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ifftshift(input, dim=None) -> Tensor

Inverse of :func:`~torch.fft.fftshift`.

Args:
    input (Tensor): the tensor in FFT order
    dim (int, Tuple[int], optional): The dimensions to rearrange.
     ...

#### torch.fft.ihfft
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ihfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the inverse of :func:`~torch.fft.hfft`.

:attr:`input` must be a real-valued signal, interpreted in the Fourier domain.
The IFF...

#### torch.fft.ihfft2
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ihfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the 2-dimensional inverse discrete Fourier transform of real
:attr:`input`. Equivalent to :func:`~torch.fft.ihfftn` but ...

#### torch.fft.ihfftn
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ihfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

Computes the N-dimensional inverse discrete Fourier transform of real :attr:`input`.

:attr:`input` must be a real-valued signal, int...

#### torch.fft.irfft
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: irfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the inverse of :func:`~torch.fft.rfft`.

:attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier
domain, as...

#### torch.fft.irfft2
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: irfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the inverse of :func:`~torch.fft.rfft2`.
Equivalent to :func:`~torch.fft.irfftn` but IFFTs only the last two dimensions ...

#### torch.fft.irfftn
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: irfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

Computes the inverse of :func:`~torch.fft.rfftn`.

:attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier
domain...

#### torch.fft.rfft
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: rfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the one dimensional Fourier transform of real-valued :attr:`input`.

The FFT of a real signal is Hermitian-symmetric, ``X[i] = c...

#### torch.fft.rfft2
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: rfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the 2-dimensional discrete Fourier transform of real :attr:`input`.
Equivalent to :func:`~torch.fft.rfftn` but FFTs only ...

#### torch.fft.rfftfreq
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: rfftfreq(n, d=1.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Computes the sample frequencies for :func:`~torch.fft.rfft` with a signal of size :attr:`...

#### torch.fft.rfftn
- **Category**: SIGNAL_PROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: rfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

Computes the N-dimensional discrete Fourier transform of real :attr:`input`.

The FFT of a real signal is Hermitian-symmetric,
``X[i_1...

#### torch.func.debug_unwrap
- **Category**: FUNCTIONAL_PROGRAMMING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Unwraps a functorch tensor (e.g. BatchedTensor, GradTrackingTensor) to its underlying tensor.

This function should only be used in a debug setting (e.g. trying to print the
value of a Tensor in a deb...

#### torch.func.functional_call
- **Category**: FUNCTIONAL_PROGRAMMING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Performs a functional call on the module by replacing the module parameters
and buffers with the provided ones.

.. note:: If the module has active parametrizations, passing a value in the
    :attr:`...

#### torch.func.functionalize
- **Category**: FUNCTIONAL_PROGRAMMING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: functionalize is a transform that can be used to remove (intermediate)
mutations and aliasing from a function, while preserving the function's
semantics.

``functionalize(func)`` returns a new functio...

#### torch.func.grad
- **Category**: FUNCTIONAL_PROGRAMMING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ``grad`` operator helps computing gradients of ``func`` with respect to the
input(s) specified by ``argnums``. This operator can be nested to
compute higher-order gradients.

Args:
    func (Callable)...

#### torch.func.grad_and_value
- **Category**: FUNCTIONAL_PROGRAMMING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns a function to compute a tuple of the gradient and primal, or
forward, computation.

Args:
    func (Callable): A Python function that takes one or more arguments.
        Must return a single-...

#### torch.func.hessian
- **Category**: FUNCTIONAL_PROGRAMMING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Computes the Hessian of ``func`` with respect to the arg(s) at index
``argnum`` via a forward-over-reverse strategy.

The forward-over-reverse strategy (composing ``jacfwd(jacrev(func))``) is
a good d...

#### torch.func.jacfwd
- **Category**: FUNCTIONAL_PROGRAMMING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Computes the Jacobian of ``func`` with respect to the arg(s) at index
``argnum`` using forward-mode autodiff

Args:
    func (function): A Python function that takes one or more arguments,
        one...

#### torch.func.jacrev
- **Category**: FUNCTIONAL_PROGRAMMING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Computes the Jacobian of ``func`` with respect to the arg(s) at index
``argnum`` using reverse mode autodiff

.. note::
    Using :attr:`chunk_size=1` is equivalent to computing the jacobian
    row-b...

#### torch.func.jvp
- **Category**: FUNCTIONAL_PROGRAMMING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Standing for the Jacobian-vector product, returns a tuple containing
the output of `func(*primals)` and the "Jacobian of ``func`` evaluated at
``primals``" times ``tangents``. This is also known as fo...

#### torch.func.linearize
- **Category**: FUNCTIONAL_PROGRAMMING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the value of ``func`` at ``primals`` and linear approximation
at ``primals``.

Args:
    func (Callable): A Python function that takes one or more arguments.
    primals (Tensors): Positional ...

#### torch.func.vjp
- **Category**: FUNCTIONAL_PROGRAMMING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Standing for the vector-Jacobian product, returns a tuple containing the
results of ``func`` applied to ``primals`` and a function that, when
given ``cotangents``, computes the reverse-mode Jacobian o...

#### torch.func.vmap
- **Category**: FUNCTIONAL_PROGRAMMING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: vmap is the vectorizing map; ``vmap(func)`` returns a new function that
maps ``func`` over some dimension of the inputs. Semantically, vmap
pushes the map into PyTorch operations called by ``func``, e...

#### torch.futures.Future
- **Category**: ASYNCHRONOUS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Wrapper around a ``torch._C.Future`` which encapsulates an asynchronous
execution of a callable, e.g. :meth:`~torch.distributed.rpc.rpc_async`. It
also exposes a set of APIs to add callback functions ...

#### torch.fx.Interpreter
- **Category**: GRAPH_TRANSFORMATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: An Interpreter executes an FX graph Node-by-Node. This pattern
can be useful for many things, including writing code
transformations as well as analysis passes.

Methods in the Interpreter class can b...

#### torch.fx.ProxyableClassMeta
- **Category**: GRAPH_TRANSFORMATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ProxyableClassMeta allows you to make construction of a given Python class
symbolically traceable. For example::

    import torch
    import torch.fx


    class TensorPair(metaclass=torch.fx.Proxyab...

#### torch.fx.Transformer
- **Category**: GRAPH_TRANSFORMATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ``Transformer`` is a special type of interpreter that produces a
new ``Module``. It exposes a ``transform()`` method that returns
the transformed ``Module``. ``Transformer`` does not require
arguments...

#### torch.hub.tqdm
- **Category**: MODEL_HUB
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Decorate an iterable object, returning an iterator which acts exactly
like the original iterable, but prints a dynamically updating
progressbar every time a value is requested.

Parameters
----------
...

#### torch.jit.Attribute
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: This method is a pass-through function that returns `value`, mostly
used to indicate to the TorchScript compiler that the left-hand side
expression is a class instance attribute with type of `type`. N...

#### torch.jit.Future
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Wrapper around a ``torch._C.Future`` which encapsulates an asynchronous
execution of a callable, e.g. :meth:`~torch.distributed.rpc.rpc_async`. It
also exposes a set of APIs to add callback functions ...

#### torch.jit.RecursiveScriptModule
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Retain the existing isinstance(ScriptModule) behavior.

The core data structure in TorchScript is the ``ScriptModule``. It is an
analogue of torch's ``nn.Module`` and represents an entire model as a t...

#### torch.jit.fork
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Create an asynchronous task executing `func` and a reference to the value of the result of this execution.

`fork` will return immediately, so the return value of `func` may not have been computed yet...

#### torch.jit.freeze
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Freeze ScriptModule, inline submodules, and attributes as constants.

Freezing a :class:`ScriptModule` will clone it and attempt to inline the cloned
module's submodules, parameters, and attributes as...

#### torch.jit.interface
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Decorate to annotate classes or modules of different types.

This decorator can be used to define an interface that can be used to annotate
classes or modules of different types. This can be used for ...

#### torch.jit.isinstance
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Provide container type refinement in TorchScript.

It can refine parameterized containers of the List, Dict, Tuple, and Optional types. E.g. ``List[str]``,
``Dict[str, List[torch.Tensor]]``, ``Optiona...

#### torch.jit.load
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Load a :class:`ScriptModule` or :class:`ScriptFunction` previously saved with :func:`torch.jit.save <torch.jit.save>`.

All previously saved modules, no matter their device, are first loaded onto CPU,...

#### torch.jit.optimize_for_inference
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Perform a set of optimization passes to optimize a model for the purposes of inference.

If the model is not already frozen, optimize_for_inference
will invoke `torch.jit.freeze` automatically.

In ad...

#### torch.jit.run_frozen_optimizations
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Run a series of optimizations looking for patterns that occur in frozen graphs.

The current set of optimizations includes:
    - Dropout Removal
    - Pretranspose Linear Layers
    - Concat Linear L...

#### torch.jit.save
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Save an offline version of this module for use in a separate process.

The saved module serializes all of the methods, submodules, parameters, and
attributes of this module. It can be loaded into the ...

#### torch.jit.save_jit_module_to_flatbuffer
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Save an offline version of this module for use in a separate process.

The saved module serializes all of the methods, submodules, parameters, and
attributes of this module. It can be loaded into the ...

#### torch.jit.script
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Script the function.

Scripting a function or ``nn.Module`` will inspect the source code, compile
it as TorchScript code using the TorchScript compiler, and return a :class:`ScriptModule` or
:class:`S...

#### torch.jit.set_fusion_strategy
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Set the type and number of specializations that can occur during fusion.

Usage: provide a list of pairs (type, depth) where type is one of "STATIC" or "DYNAMIC"
and depth is an integer.

Behavior - s...

#### torch.jit.trace
- **Category**: JIT_COMPILATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Trace a function and return an executable  or :class:`ScriptFunction` that will be optimized using just-in-time compilation.

Tracing is ideal for code that operates only on
``Tensor``\\s and lists, d...

#### torch.linalg.LinAlgError
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Error raised by torch.linalg function when the cause of error is a numerical inconsistency in the data.
 For example, you can the torch.linalg.inv function will raise torch.linalg.LinAlgError when it ...

#### torch.linalg.cholesky
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.cholesky(A, *, upper=False, out=None) -> Tensor

Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.

Letting :math:`\mathbb{K}` be :math:`\ma...

#### torch.linalg.cholesky_ex
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.cholesky_ex(A, *, upper=False, check_errors=False, out=None) -> (Tensor, Tensor)

Computes the Cholesky decomposition of a complex Hermitian or real
symmetric positive-definite matrix.

This fu...

#### torch.linalg.cond
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.cond(A, p=None, *, out=None) -> Tensor

Computes the condition number of a matrix with respect to a matrix norm.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **c...

#### torch.linalg.cross
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.cross(input, other, *, dim=-1, out=None) -> Tensor


Computes the cross product of two 3-dimensional vectors.

Supports input of float, double, cfloat and cdouble dtypes. Also supports batches
...

#### torch.linalg.det
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.det(A, *, out=None) -> Tensor

Computes the determinant of a square matrix.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a...

#### torch.linalg.diagonal
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.diagonal(A, *, offset=0, dim1=-2, dim2=-1) -> Tensor

Alias for :func:`torch.diagonal` with defaults :attr:`dim1`\ `= -2`, :attr:`dim2`\ `= -1`....

#### torch.linalg.eig
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.eig(A, *, out=None) -> (Tensor, Tensor)

Computes the eigenvalue decomposition of a square matrix if it exists.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **ei...

#### torch.linalg.eigh
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.eigh(A, UPLO='L', *, out=None) -> (Tensor, Tensor)

Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :...

#### torch.linalg.eigvals
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.eigvals(A, *, out=None) -> Tensor

Computes the eigenvalues of a square matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalues** of a square matrix :...

#### torch.linalg.eigvalsh
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.eigvalsh(A, UPLO='L', *, out=None) -> Tensor

Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
...

#### torch.linalg.householder_product
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: householder_product(A, tau, *, out=None) -> Tensor

Computes the first `n` columns of a product of Householder matrices.

Let :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, and
let :m...

#### torch.linalg.inv
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.inv(A, *, out=None) -> Tensor

Computes the inverse of a square matrix if it exists.
Throws a `RuntimeError` if the matrix is not invertible.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` o...

#### torch.linalg.inv_ex
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.inv_ex(A, *, check_errors=False, out=None) -> (Tensor, Tensor)

Computes the inverse of a square matrix if it is invertible.

Returns a namedtuple ``(inverse, info)``. ``inverse`` contains the ...

#### torch.linalg.ldl_factor
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.ldl_factor(A, *, hermitian=False, out=None) -> (Tensor, Tensor)

Computes a compact representation of the LDL factorization of a Hermitian or symmetric (possibly indefinite) matrix.

When :attr...

#### torch.linalg.ldl_factor_ex
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.ldl_factor_ex(A, *, hermitian=False, check_errors=False, out=None) -> (Tensor, Tensor, Tensor)

This is a version of :func:`~ldl_factor` that does not perform error checks unless :attr:`check_e...

#### torch.linalg.ldl_solve
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.ldl_solve(LD, pivots, B, *, hermitian=False, out=None) -> Tensor

Computes the solution of a system of linear equations using the LDL factorization.

:attr:`LD` and :attr:`pivots` are the compa...

#### torch.linalg.lstsq
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: torch.linalg.lstsq(A, B, rcond=None, *, driver=None) -> (Tensor, Tensor, Tensor, Tensor)

Computes a solution to the least squares problem of a system of linear equations.

Letting :math:`\mathbb{K}` ...

#### torch.linalg.lu
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: lu(A, *, pivot=True, out=None) -> (Tensor, Tensor, Tensor)

Computes the LU decomposition with partial pivoting of a matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
th...

#### torch.linalg.lu_factor
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.lu_factor(A, *, bool pivot=True, out=None) -> (Tensor, Tensor)

Computes a compact representation of the LU factorization with partial pivoting of a matrix.

This function computes a compact re...

#### torch.linalg.lu_factor_ex
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.lu_factor_ex(A, *, pivot=True, check_errors=False, out=None) -> (Tensor, Tensor, Tensor)

This is a version of :func:`~lu_factor` that does not perform error checks unless :attr:`check_errors`\...

#### torch.linalg.lu_solve
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.lu_solve(LU, pivots, B, *, left=True, adjoint=False, out=None) -> Tensor

Computes the solution of a square system of linear equations with a unique solution given an LU decomposition.

Letting...

#### torch.linalg.matmul
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.matmul(input, other, *, out=None) -> Tensor

Alias for :func:`torch.matmul`...

#### torch.linalg.matrix_exp
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.matrix_exp(A) -> Tensor

Computes the matrix exponential of a square matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
this function computes the **matrix exponen...

#### torch.linalg.matrix_norm
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None, out=None) -> Tensor

Computes a matrix norm.

If :attr:`A` is complex valued, it computes the norm of :attr:`A`\ `.abs()`

...

#### torch.linalg.matrix_power
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: matrix_power(A, n, *, out=None) -> Tensor

Computes the `n`-th power of a square matrix for an integer `n`.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matric...

#### torch.linalg.matrix_rank
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None) -> Tensor

Computes the numerical rank of a matrix.

The matrix rank is computed as the number of singular values
(or eigenval...

#### torch.linalg.multi_dot
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.multi_dot(tensors, *, out=None)

Efficiently multiplies two or more matrices by reordering the multiplications so that
the fewest arithmetic operations are performed.

Supports inputs of float,...

#### torch.linalg.norm
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None) -> Tensor

Computes a vector or matrix norm.

Supports input of float, double, cfloat and cdouble dtypes.

Whether this funct...

#### torch.linalg.pinv
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.pinv(A, *, atol=None, rtol=None, hermitian=False, out=None) -> Tensor

Computes the pseudoinverse (Moore-Penrose inverse) of a matrix.

The pseudoinverse may be `defined algebraically`_
but it ...

#### torch.linalg.qr
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: qr(A, mode='reduced', *, out=None) -> (Tensor, Tensor)

Computes the QR decomposition of a matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **full QR decomposition*...

#### torch.linalg.slogdet
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.slogdet(A, *, out=None) -> (Tensor, Tensor)

Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.

For complex :attr:`A`, it returns the sign and...

#### torch.linalg.solve
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.solve(A, B, *, left=True, out=None) -> Tensor

Computes the solution of a square system of linear equations with a unique solution.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\...

#### torch.linalg.solve_ex
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.solve_ex(A, B, *, left=True, check_errors=False, out=None) -> (Tensor, Tensor)

A version of :func:`~solve` that does not perform error checks unless :attr:`check_errors`\ `= True`.
It also ret...

#### torch.linalg.solve_triangular
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None) -> Tensor

Computes the solution of a triangular system of linear equations with a unique solution.

Letting :math:`\m...

#### torch.linalg.svd
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.svd(A, full_matrices=True, *, driver=None, out=None) -> (Tensor, Tensor, Tensor)

Computes the singular value decomposition (SVD) of a matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` ...

#### torch.linalg.svdvals
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.svdvals(A, *, driver=None, out=None) -> Tensor

Computes the singular values of a matrix.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if ...

#### torch.linalg.vander
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: vander(x, N=None) -> Tensor

Generates a Vandermonde matrix.

Returns the Vandermonde matrix :math:`V`

.. math::

    V = \begin{pmatrix}
            1 & x_1 & x_1^2 & \dots & x_1^{N-1}\\
           ...

#### torch.linalg.vecdot
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.vecdot(x, y, *, dim=-1, out=None) -> Tensor

Computes the dot product of two batches of vectors along a dimension.

In symbols, this function computes

.. math::

    \sum_{i=1}^n \overline{x_i...

#### torch.linalg.vector_norm
- **Category**: LINEAR_ALGEBRA
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: linalg.vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor

Computes a vector norm.

If :attr:`x` is complex valued, it computes the norm of :attr:`x`\ `.abs()`

Supports...

#### torch.masked.amax
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: amax(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns maximum of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
el...

#### torch.masked.amin
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: amin(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns minimum of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
el...

#### torch.masked.argmax
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: argmax(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns argmax of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
el...

#### torch.masked.argmin
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: argmin(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns argmin of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
el...

#### torch.masked.cumprod
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cumprod(input, dim, *, dtype=None, mask=None) -> Tensor

Returns cumulative_prod of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
accordi...

#### torch.masked.cumsum
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: cumsum(input, dim, *, dtype=None, mask=None) -> Tensor

Returns cumulative_sum of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according...

#### torch.masked.log_softmax
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: log_softmax(input, dim, *, dtype=None, mask=None) -> Tensor

Returns log_softmax of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
accordi...

#### torch.masked.logaddexp
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logaddexp(input, other, *, dtype=None, input_mask=None, other_mask=None) -> Tensor

Returns logaddexp of all the elements in the :attr:`input` and the :attr:`other`
tensor. The :attr:`input` elements ...

#### torch.masked.logsumexp
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logsumexp(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns logsumexp of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`in...

#### torch.masked.mean
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: mean(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns mean of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
eleme...

#### torch.masked.median
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: median(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns median of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
el...

#### torch.masked.norm
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: norm(input, ord, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns norm of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
...

#### torch.masked.normalize
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: normalize(input, ord, dim, *, eps=1e-12, dtype=None, mask=None) -> Tensor

Returns normalize of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked...

#### torch.masked.prod
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: prod(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns product of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
el...

#### torch.masked.softmax
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: softmax(input, dim, *, dtype=None, mask=None) -> Tensor

Returns softmax of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to th...

#### torch.masked.softmin
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: softmin(input, dim, *, dtype=None, mask=None) -> Tensor

Returns softmin of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to th...

#### torch.masked.std
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: std(input, dim, unbiased, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns standard_deviation of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while t...

#### torch.masked.sum
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sum(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns sum of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
element...

#### torch.masked.var
- **Category**: MASKED_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: var(input, dim, unbiased, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns variance of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`...

#### torch.mps.compile_shader
- **Category**: MPS_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Compiles compute shader from source and allows one to invoke kernels
defined there from the comfort of Python runtime
Example::

    >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_MPS)
    >>> lib = torc...

#### torch.mps.empty_cache
- **Category**: MPS_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Releases all unoccupied cached memory currently held by the caching
allocator so that those can be used in other GPU applications....

#### torch.mps.get_rng_state
- **Category**: MPS_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the random number generator state as a ByteTensor.

Args:
    device (torch.device or int, optional): The device to return the RNG state of.
        Default: ``'mps'`` (i.e., ``torch.device('m...

#### torch.mps.is_available
- **Category**: MPS_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ...

#### torch.mps.manual_seed
- **Category**: MPS_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the seed for generating random numbers.

Args:
    seed (int): The desired seed....

#### torch.mps.seed
- **Category**: MPS_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the seed for generating random numbers to a random number....

#### torch.mps.set_rng_state
- **Category**: MPS_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the random number generator state.

Args:
    new_state (torch.ByteTensor): The desired state
    device (torch.device or int, optional): The device to set the RNG state.
        Default: ``'mps'...

#### torch.mps.synchronize
- **Category**: MPS_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Waits for all kernels in all streams on a MPS device to complete....

#### torch.mtia.empty_cache
- **Category**: MTIA_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Empty the MTIA device cache....

#### torch.mtia.get_rng_state
- **Category**: MTIA_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the random number generator state as a ByteTensor.

Args:
    device (torch.device or int, optional): The device to return the RNG state of.
        Default: ``'mtia'`` (i.e., ``torch.device('...

#### torch.mtia.is_available
- **Category**: MTIA_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return true if MTIA device is available...

#### torch.mtia.set_rng_state
- **Category**: MTIA_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the random number generator state.

Args:
    new_state (torch.ByteTensor): The desired state
    device (torch.device or int, optional): The device to set the RNG state.
        Default: ``'mtia...

#### torch.mtia.synchronize
- **Category**: MTIA_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Waits for all jobs in all streams on a MTIA device to complete....

#### torch.multiprocessing.get_sharing_strategy
- **Category**: MULTIPROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return the current strategy for sharing CPU tensors....

#### torch.multiprocessing.set_sharing_strategy
- **Category**: MULTIPROCESSING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Set the strategy for sharing CPU tensors.

Args:
    new_strategy (str): Name of the selected strategy. Should be one of
        the values returned by :func:`get_all_sharing_strategies()`....

#### torch.nested.masked_select
- **Category**: NESTED_TENSORS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Constructs a nested tensor given a strided tensor input and a strided mask, the resulting jagged layout nested tensor
will have values retain values where the mask is equal to True. The dimensionality...

#### torch.nested.narrow
- **Category**: NESTED_TENSORS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Constructs a nested tensor (which might be a view) from :attr:`tensor`, a strided tensor. This follows
similar semantics to torch.Tensor.narrow, where in the :attr:`dim`-th dimension the new nested te...

#### torch.optim.ASGD
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Implements Averaged Stochastic Gradient Descent.

It has been proposed in `Acceleration of stochastic approximation by
averaging`_.

Args:
    params (iterable): iterable of parameters or named_parame...

#### torch.optim.Adadelta
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Implements Adadelta algorithm.

.. math::
   \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{input}      : \gamma \text...

#### torch.optim.Adafactor
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Implements Adafactor algorithm.

.. math::
    \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{input}      : \gamma \te...

#### torch.optim.Adagrad
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Implements Adagrad algorithm.

.. math::
   \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{input}      : \gamma \text{...

#### torch.optim.Adam
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Implements Adam algorithm.

.. math::
   \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{input}      : \gamma \text{ (l...

#### torch.optim.AdamW
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Implements AdamW algorithm, where weight decay does not accumulate in the momentum nor variance.

.. math::
   \begin{aligned}
        &\rule{110mm}{0.4pt}                                             ...

#### torch.optim.Adamax
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Implements Adamax algorithm (a variant of Adam based on infinity norm).

.. math::
   \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
  ...

#### torch.optim.LBFGS
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Implements L-BFGS algorithm.

Heavily inspired by `minFunc
<https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

.. warning::
    This optimizer doesn't support per-parameter options and paramet...

#### torch.optim.NAdam
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Implements NAdam algorithm.

.. math::
   \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{input}      : \gamma_t \text{...

#### torch.optim.Optimizer
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Base class for all optimizers.

.. warning::
    Parameters need to be specified as collections that have a deterministic
    ordering that is consistent between runs. Examples of objects that don't
 ...

#### torch.optim.RAdam
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Implements RAdam algorithm.

.. math::
   \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{input}      : \gamma \text{ (...

#### torch.optim.RMSprop
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Implements RMSprop algorithm.

.. math::
   \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{input}      : \alpha \text{...

#### torch.optim.Rprop
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Implements the resilient backpropagation algorithm.

.. math::
   \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{input...

#### torch.optim.SGD
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Implements stochastic gradient descent (optionally with momentum).

.. math::
   \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
       ...

#### torch.optim.SparseAdam
- **Category**: OPTIMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: SparseAdam implements a masked version of the Adam algorithm
suitable for sparse gradients. Currently, due to implementation constraints (explained
below), SparseAdam is only intended for a narrow sub...

#### torch.overrides.BaseTorchFunctionMode
- **Category**: CUSTOMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: A ``TorchFunctionMode`` allows you to override the meaning of all
``__torch_function__`` overrideable functions within a dynamic scope,
without having to actually create a tensor subclass or manually
...

#### torch.overrides.TorchFunctionMode
- **Category**: CUSTOMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: A ``TorchFunctionMode`` allows you to override the meaning of all
``__torch_function__`` overrideable functions within a dynamic scope,
without having to actually create a tensor subclass or manually
...

#### torch.overrides.has_torch_function
- **Category**: CUSTOMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Check for __torch_function__ implementations in the elements of an iterable
or if a __torch_function__ mode is enabled.  Considers exact ``Tensor`` s
and ``Parameter`` s non-dispatchable.  Use this to...

#### torch.overrides.wrap_torch_function
- **Category**: CUSTOMIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Wraps a given function with ``__torch_function__`` -related functionality.

Parameters
----------
dispatcher: Callable
    A callable that returns an iterable of Tensor-likes passed into the function....

#### torch.profiler.ProfilerActivity
- **Category**: PROFILING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Members:

CPU

XPU

MTIA

CUDA

HPU

PrivateUse1...

#### torch.profiler.profile
- **Category**: PROFILING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Profiler context manager.

Args:
    activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
        ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler...

#### torch.profiler.record_function
- **Category**: PROFILING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Context manager/function decorator that adds a label to a code block/function when running autograd profiler.
Label will only appear if CPU activity tracing is enabled.

It is useful when tracing the ...

#### torch.profiler.supported_activities
- **Category**: PROFILING
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns a set of supported profiler tracing activities.

Note: profiler uses CUPTI library to trace on-device CUDA kernels.
In case when CUDA is enabled but CUPTI is not available, passing
``ProfilerA...

#### torch.quantization.DeQuantStub
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Dequantize stub module, before calibration, this is same as identity,
this will be swapped as `nnq.DeQuantize` in `convert`.

Args:
    qconfig: quantization configuration for the tensor,
        if q...

#### torch.quantization.FakeQuantize
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Simulate the quantize and dequantize operations in training time.

The output of this module is given by::

    x_out = (
      clamp(round(x/scale + zero_point), quant_min, quant_max) - zero_point
  ...

#### torch.quantization.FakeQuantizeBase
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Base fake quantize module.

Base fake quantize module
Any fake quantize implementation should derive from this class.

Concrete fake quantize module should follow the same API. In forward, they will u...

#### torch.quantization.FixedQParamsFakeQuantize
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Simulate quantize and dequantize in training time.

Simulate quantize and dequantize with fixed quantization
parameters in training time. Only per tensor quantization
is supported....

#### torch.quantization.FusedMovingAvgObsFakeQuantize
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Define a fused module to observe the tensor.

Fused module that is used to observe the input tensor (compute min/max), compute
scale/zero_point and fake_quantize the tensor.
This module uses calculati...

#### torch.quantization.HistogramObserver
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: The module records the running histogram of tensor values along with
min/max values. ``calculate_qparams`` will calculate scale and zero_point.

Args:
    bins: Number of bins to use for the histogram...

#### torch.quantization.MinMaxObserver
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Observer module for computing the quantization parameters based on the
running min and max values.

This observer uses the tensor min/max statistics to compute the quantization
parameters. The module ...

#### torch.quantization.MovingAverageMinMaxObserver
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Observer module for computing the quantization parameters based on the
moving average of the min and max values.

This observer computes the quantization parameters based on the moving
averages of min...

#### torch.quantization.ObserverBase
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Base observer Module.
Any observer implementation should derive from this class.

Concrete observers should follow the same API. In forward, they will update
the statistics of the observed Tensor. And...

#### torch.quantization.QuantStub
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Quantize stub module, before calibration, this is same as an observer,
it will be swapped as `nnq.Quantize` in `convert`.

Args:
    qconfig: quantization configuration for the tensor,
        if qcon...

#### torch.quantization.QuantWrapper
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: A wrapper class that wraps the input module, adds QuantStub and
DeQuantStub and surround the call to module with call to quant and dequant
modules.

This is used by the `quantization` utility function...

#### torch.quantization.RecordingObserver
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: The module is mainly for debug and records the tensor values during runtime.

Args:
    dtype: Quantized data type
    qscheme: Quantization scheme to be used
    reduce_range: Reduces the range of th...

#### torch.quantization.default_debug_observer
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: The module is mainly for debug and records the tensor values during runtime.

Args:
    dtype: Quantized data type
    qscheme: Quantization scheme to be used
    reduce_range: Reduces the range of th...

#### torch.quantization.default_eval_fn
- **Category**: QUANTIZATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Default evaluation function takes a torch.utils.data.Dataset or a list of
input Tensors and run the model on the dataset...

#### torch.random.fork_rng
- **Category**: RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Forks the RNG, so that when you return, the RNG is reset
to the state that it was previously in.

Args:
    devices (iterable of Device IDs): devices for which to fork
        the RNG. CPU RNG state i...

#### torch.random.get_rng_state
- **Category**: RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the random number generator state as a `torch.ByteTensor`.

.. note:: The returned state is for the default generator on CPU only.

See also: :func:`torch.random.fork_rng`....

#### torch.random.initial_seed
- **Category**: RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Returns the initial seed for generating random numbers as a
Python `long`.

.. note:: The returned seed is for the default generator on CPU only....

#### torch.random.manual_seed
- **Category**: RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the seed for generating random numbers on all devices. Returns a
`torch.Generator` object.

Args:
    seed (int): The desired seed. Value must be within the inclusive range
        `[-0x8000_0000...

#### torch.random.seed
- **Category**: RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the seed for generating random numbers to a non-deterministic
random number on all devices. Returns a 64 bit number used to seed the RNG....

#### torch.random.set_rng_state
- **Category**: RANDOM_GENERATION
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Sets the random number generator state.

.. note:: This function only works for CPU. For CUDA, please use
    :func:`torch.manual_seed`, which works for both CPU and CUDA.

Args:
    new_state (torch....

#### torch.sparse.addmm
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sparse.addmm(mat, mat1, mat2, *, beta=1., alpha=1.) -> Tensor

This function does exact same thing as :func:`torch.addmm` in the forward,
except that it supports backward for sparse COO matrix :attr:`...

#### torch.sparse.as_sparse_gradcheck
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Decorate function, to extend gradcheck for sparse tensors.

Decorator for torch.autograd.gradcheck or its functools.partial
variants that extends the gradcheck function with support to input
functions...

#### torch.sparse.log_softmax
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sparse.log_softmax(input, dim, *, dtype=None) -> Tensor

Applies a softmax function followed by logarithm.

See :class:`~torch.sparse.softmax` for more details.

Args:
    input (Tensor): input
    di...

#### torch.sparse.mm
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**:     Performs a matrix multiplication of the sparse matrix :attr:`mat1`
    and the (sparse or strided) matrix :attr:`mat2`. Similar to :func:`torch.mm`, if :attr:`mat1` is a
    :math:`(n \times m)` t...

#### torch.sparse.sampled_addmm
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sparse.sampled_addmm(input, mat1, mat2, *, beta=1., alpha=1., out=None) -> Tensor

Performs a matrix multiplication of the dense matrices :attr:`mat1` and :attr:`mat2` at the locations
specified by th...

#### torch.sparse.softmax
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sparse.softmax(input, dim, *, dtype=None) -> Tensor

Applies a softmax function.

Softmax is defined as:

:math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}`

where :math:`i, j` run over ...

#### torch.sparse.spdiags
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sparse.spdiags(diagonals, offsets, shape, layout=None) -> Tensor

Creates a sparse 2D tensor by placing the values from rows of
:attr:`diagonals` along specified diagonals of the output

The :attr:`of...

#### torch.sparse.spsolve
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sparse.spsolve(input, other, *, left=True) -> Tensor

Computes the solution of a square system of linear equations with
a unique solution. Its purpose is similar to :func:`torch.linalg.solve`,
except ...

#### torch.sparse.sum
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return the sum of each row of the given sparse tensor.

Returns the sum of each row of the sparse tensor :attr:`input` in the given
dimensions :attr:`dim`. If :attr:`dim` is a list of dimensions,
redu...

#### torch.sparse.to_sparse_semi_structured
- **Category**: SPARSE_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: This function converts a dense tensor into a sparse semi-structured tensor.
It will return a SparseSemiStructuredTensor, a subclass of torch.Tensor.

This function will check to ensure the dense tenso...

#### torch.special.airy_ai
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: airy_ai(input, *, out=None) -> Tensor

Airy function :math:`\text{Ai}\left(\text{input}\right)`.


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output ten...

#### torch.special.bessel_j0
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bessel_j0(input, *, out=None) -> Tensor

Bessel function of the first kind of order :math:`0`.


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tenso...

#### torch.special.bessel_j1
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bessel_j1(input, *, out=None) -> Tensor

Bessel function of the first kind of order :math:`1`.


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tenso...

#### torch.special.bessel_y0
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bessel_y0(input, *, out=None) -> Tensor

Bessel function of the second kind of order :math:`0`.


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tens...

#### torch.special.bessel_y1
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: bessel_y1(input, *, out=None) -> Tensor

Bessel function of the second kind of order :math:`1`.


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tens...

#### torch.special.chebyshev_polynomial_t
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: chebyshev_polynomial_t(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the first kind :math:`T_{n}(\text{input})`.

If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{in...

#### torch.special.chebyshev_polynomial_u
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: chebyshev_polynomial_t(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the second kind :math:`U_{n}(\text{input})`.

If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`,
:math:`2 \time...

#### torch.special.chebyshev_polynomial_v
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: chebyshev_polynomial_v(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the third kind :math:`V_{n}^{\ast}(\text{input})`.


Args:
    input (Tensor): the input tensor.
    n (Tensor): Degree...

#### torch.special.chebyshev_polynomial_w
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: chebyshev_polynomial_w(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the fourth kind :math:`W_{n}^{\ast}(\text{input})`.


Args:
    input (Tensor): the input tensor.
    n (Tensor): Degre...

#### torch.special.digamma
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: digamma(input, *, out=None) -> Tensor

Computes the logarithmic derivative of the gamma function on `input`.

.. math::
    \digamma(x) = \frac{d}{dx} \ln\left(\Gamma\left(x\right)\right) = \frac{\Gam...

#### torch.special.entr
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: entr(input, *, out=None) -> Tensor
Computes the entropy on :attr:`input` (as defined below), elementwise.

.. math::
    \begin{align}
    \text{entr(x)} = \begin{cases}
        -x * \ln(x)  & x > 0 \...

#### torch.special.erf
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: erf(input, *, out=None) -> Tensor

Computes the error function of :attr:`input`. The error function is defined as follows:

.. math::
    \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} d...

#### torch.special.erfc
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: erfc(input, *, out=None) -> Tensor

Computes the complementary error function of :attr:`input`.
The complementary error function is defined as follows:

.. math::
    \mathrm{erfc}(x) = 1 - \frac{2}{\...

#### torch.special.erfcx
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: erfcx(input, *, out=None) -> Tensor

Computes the scaled complementary error function for each element of :attr:`input`.
The scaled complementary error function is defined as follows:

.. math::
    \...

#### torch.special.erfinv
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: erfinv(input, *, out=None) -> Tensor

Computes the inverse error function of :attr:`input`.
The inverse error function is defined in the range :math:`(-1, 1)` as:

.. math::
    \mathrm{erfinv}(\mathr...

#### torch.special.exp2
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: exp2(input, *, out=None) -> Tensor

Computes the base two exponential function of :attr:`input`.

.. math::
    y_{i} = 2^{x_{i}}


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (...

#### torch.special.expit
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: expit(input, *, out=None) -> Tensor

Computes the expit (also known as the logistic sigmoid function) of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{1}{1 + e^{-\text{input}_{i...

#### torch.special.expm1
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: expm1(input, *, out=None) -> Tensor

Computes the exponential of the elements minus 1
of :attr:`input`.

.. math::
    y_{i} = e^{x_{i}} - 1

.. note:: This function provides greater precision than ex...

#### torch.special.gammainc
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: gammainc(input, other, *, out=None) -> Tensor

Computes the regularized lower incomplete gamma function:

.. math::
    \text{out}_{i} = \frac{1}{\Gamma(\text{input}_i)} \int_0^{\text{other}_i} t^{\te...

#### torch.special.gammaincc
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: gammaincc(input, other, *, out=None) -> Tensor

Computes the regularized upper incomplete gamma function:

.. math::
    \text{out}_{i} = \frac{1}{\Gamma(\text{input}_i)} \int_{\text{other}_i}^{\infty...

#### torch.special.gammaln
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: gammaln(input, *, out=None) -> Tensor

Computes the natural logarithm of the absolute value of the gamma function on :attr:`input`.

.. math::
    \text{out}_{i} = \ln \Gamma(|\text{input}_{i}|)

Args...

#### torch.special.hermite_polynomial_h
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: hermite_polynomial_h(input, n, *, out=None) -> Tensor

Physicist's Hermite polynomial :math:`H_{n}(\text{input})`.

If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{input}`
is r...

#### torch.special.hermite_polynomial_he
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: hermite_polynomial_he(input, n, *, out=None) -> Tensor

Probabilist's Hermite polynomial :math:`He_{n}(\text{input})`.

If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{input}`
...

#### torch.special.i0
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: i0(input, *, out=None) -> Tensor

Computes the zeroth order modified Bessel function of the first kind for each element of :attr:`input`.

.. math::
    \text{out}_{i} = I_0(\text{input}_{i}) = \sum_{...

#### torch.special.i0e
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: i0e(input, *, out=None) -> Tensor
Computes the exponentially scaled zeroth order modified Bessel function of the first kind (as defined below)
for each element of :attr:`input`.

.. math::
    \text{o...

#### torch.special.i1
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: i1(input, *, out=None) -> Tensor
Computes the first order modified Bessel function of the first kind (as defined below)
for each element of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{(\text{...

#### torch.special.i1e
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: i1e(input, *, out=None) -> Tensor
Computes the exponentially scaled first order modified Bessel function of the first kind (as defined below)
for each element of :attr:`input`.

.. math::
    \text{ou...

#### torch.special.laguerre_polynomial_l
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: laguerre_polynomial_l(input, n, *, out=None) -> Tensor

Laguerre polynomial :math:`L_{n}(\text{input})`.

If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{input}`
is returned. O...

#### torch.special.legendre_polynomial_p
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: legendre_polynomial_p(input, n, *, out=None) -> Tensor

Legendre polynomial :math:`P_{n}(\text{input})`.

If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{input}`
is returned. O...

#### torch.special.log1p
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: log1p(input, *, out=None) -> Tensor

Alias for :func:`torch.log1p`....

#### torch.special.log_ndtr
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: log_ndtr(input, *, out=None) -> Tensor
Computes the log of the area under the standard Gaussian probability density function,
integrated from minus infinity to :attr:`input`, elementwise.

.. math::
 ...

#### torch.special.log_softmax
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: log_softmax(input, dim, *, dtype=None) -> Tensor

Computes softmax followed by a logarithm.

While mathematically equivalent to log(softmax(x)), doing these two
operations separately is slower and num...

#### torch.special.logit
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: logit(input, eps=None, *, out=None) -> Tensor

Returns a new tensor with the logit of the elements of :attr:`input`.
:attr:`input` is clamped to [eps, 1 - eps] when eps is not None.
When eps is None a...

#### torch.special.modified_bessel_i0
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: modified_bessel_i0(input, *, out=None) -> Tensor

Modified Bessel function of the first kind of order :math:`0`.


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional)...

#### torch.special.modified_bessel_i1
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: modified_bessel_i1(input, *, out=None) -> Tensor

Modified Bessel function of the first kind of order :math:`1`.


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional)...

#### torch.special.modified_bessel_k0
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: modified_bessel_k0(input, *, out=None) -> Tensor

Modified Bessel function of the second kind of order :math:`0`.


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional...

#### torch.special.modified_bessel_k1
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: modified_bessel_k1(input, *, out=None) -> Tensor

Modified Bessel function of the second kind of order :math:`1`.


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional...

#### torch.special.multigammaln
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: multigammaln(input, p, *, out=None) -> Tensor

Computes the `multivariate log-gamma function
<https://en.wikipedia.org/wiki/Multivariate_gamma_function>`_ with dimension
:math:`p` element-wise, given ...

#### torch.special.ndtr
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ndtr(input, *, out=None) -> Tensor
Computes the area under the standard Gaussian probability density function,
integrated from minus infinity to :attr:`input`, elementwise.

.. math::
    \text{ndtr}(...

#### torch.special.ndtri
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: ndtri(input, *, out=None) -> Tensor
Computes the argument, x, for which the area under the Gaussian probability density function
(integrated from minus infinity to x) is equal to :attr:`input`, elemen...

#### torch.special.polygamma
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: polygamma(n, input, *, out=None) -> Tensor

Computes the :math:`n^{th}` derivative of the digamma function on :attr:`input`.
:math:`n \geq 0` is called the order of the polygamma function.

.. math::
...

#### torch.special.psi
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: psi(input, *, out=None) -> Tensor

Alias for :func:`torch.special.digamma`....

#### torch.special.round
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: round(input, *, out=None) -> Tensor

Alias for :func:`torch.round`....

#### torch.special.scaled_modified_bessel_k0
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: scaled_modified_bessel_k0(input, *, out=None) -> Tensor

Scaled modified Bessel function of the second kind of order :math:`0`.


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Te...

#### torch.special.scaled_modified_bessel_k1
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: scaled_modified_bessel_k1(input, *, out=None) -> Tensor

Scaled modified Bessel function of the second kind of order :math:`1`.


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Te...

#### torch.special.shifted_chebyshev_polynomial_t
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: shifted_chebyshev_polynomial_t(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the first kind :math:`T_{n}^{\ast}(\text{input})`.


Args:
    input (Tensor): the input tensor.
    n (Tensor)...

#### torch.special.shifted_chebyshev_polynomial_u
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: shifted_chebyshev_polynomial_u(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the second kind :math:`U_{n}^{\ast}(\text{input})`.


Args:
    input (Tensor): the input tensor.
    n (Tensor...

#### torch.special.shifted_chebyshev_polynomial_v
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: shifted_chebyshev_polynomial_v(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the third kind :math:`V_{n}^{\ast}(\text{input})`.


Args:
    input (Tensor): the input tensor.
    n (Tensor)...

#### torch.special.shifted_chebyshev_polynomial_w
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: shifted_chebyshev_polynomial_w(input, n, *, out=None) -> Tensor

Chebyshev polynomial of the fourth kind :math:`W_{n}^{\ast}(\text{input})`.


Args:
    input (Tensor): the input tensor.
    n (Tensor...

#### torch.special.sinc
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: sinc(input, *, out=None) -> Tensor

Computes the normalized sinc of :attr:`input.`

.. math::
    \text{out}_{i} =
    \begin{cases}
      1, & \text{if}\ \text{input}_{i}=0 \\
      \sin(\pi \text{in...

#### torch.special.softmax
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: softmax(input, dim, *, dtype=None) -> Tensor

Computes the softmax function.

Softmax is defined as:

:math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

It is applied to all slices al...

#### torch.special.spherical_bessel_j0
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: spherical_bessel_j0(input, *, out=None) -> Tensor

Spherical Bessel function of the first kind of order :math:`0`.


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optiona...

#### torch.special.xlog1py
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: xlog1py(input, other, *, out=None) -> Tensor

Computes ``input * log1p(other)`` with the following cases.

.. math::
    \text{out}_{i} = \begin{cases}
        \text{NaN} & \text{if } \text{other}_{i}...

#### torch.special.xlogy
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: xlogy(input, other, *, out=None) -> Tensor

Computes ``input * log(other)`` with the following cases.

.. math::
    \text{out}_{i} = \begin{cases}
        \text{NaN} & \text{if } \text{other}_{i} = \...

#### torch.special.zeta
- **Category**: SPECIAL_FUNCTIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: zeta(input, other, *, out=None) -> Tensor

Computes the Hurwitz zeta function, elementwise.

.. math::
    \zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x}


Args:
    input (Tensor): the input t...

#### torch.testing.assert_close
- **Category**: TESTING_UTILITIES
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Asserts that ``actual`` and ``expected`` are close.

If ``actual`` and ``expected`` are strided, non-quantized, real-valued, and finite, they are considered close if

.. math::

    \lvert \text{actua...

#### torch.types.DispatchKey
- **Category**: TYPE_SYSTEM
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Members:

Undefined

CompositeExplicitAutogradNonFunctional

CompositeExplicitAutograd

CompositeImplicitAutogradNestedTensor

CompositeImplicitAutograd

AutogradNestedTensor

AutogradOther

Autograd
...

#### torch.utils.generate_methods_for_privateuse1_backend
- **Category**: UTILITIES
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Automatically generate attributes and methods for the custom backend after rename privateuse1 backend.

In the default scenario, storage-related methods will not be generated automatically.

When you ...

#### torch.utils.rename_privateuse1_backend
- **Category**: UTILITIES
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Rename the privateuse1 backend device to make it more convenient to use as a device name within PyTorch APIs.

The steps are:

(1) (In C++) implement kernels for various torch operations, and register...

#### torch.xpu.Any
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements...

#### torch.xpu.empty_cache
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Release all unoccupied cached memory currently held by the caching
allocator so that those can be used in other XPU application.

.. note::
    :func:`~torch.xpu.empty_cache` doesn't increase the amou...

#### torch.xpu.get_arch_list
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return list XPU architectures this library was compiled for....

#### torch.xpu.get_gencode_flags
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return XPU AOT(ahead-of-time) build flags this library was compiled with....

#### torch.xpu.get_rng_state
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return the random number generator state of the specified GPU as a ByteTensor.

Args:
    device (torch.device or int, optional): The device to return the RNG state of.
        Default: ``'xpu'`` (i.e...

#### torch.xpu.get_rng_state_all
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return a list of ByteTensor representing the random number states of all devices....

#### torch.xpu.init
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Initialize PyTorch's XPU state.
This is a Python API about lazy initialization that avoids initializing
XPU until the first time it is accessed. Does nothing if the XPU state is
already initialized....

#### torch.xpu.initial_seed
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return the current random seed of the current GPU.

.. warning::
    This function eagerly initializes XPU....

#### torch.xpu.is_available
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return a bool indicating if XPU is currently available....

#### torch.xpu.is_bf16_supported
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return a bool indicating if the current XPU device supports dtype bfloat16....

#### torch.xpu.is_initialized
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return whether PyTorch's XPU state has been initialized....

#### torch.xpu.lru_cache
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Least-recently-used cache decorator.

If *maxsize* is set to None, the LRU features are disabled and the cache
can grow without bound.

If *typed* is True, arguments of different types will be cached ...

#### torch.xpu.manual_seed
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Set the seed for generating random numbers for the current GPU.

It's safe to call this function if XPU is not available; in that case, it is silently ignored.

Args:
    seed (int): The desired seed....

#### torch.xpu.manual_seed_all
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Set the seed for generating random numbers on all GPUs.

It's safe to call this function if XPU is not available; in that case, it is silently ignored.

Args:
    seed (int): The desired seed....

#### torch.xpu.mem_get_info
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Return the global free and total GPU memory for a given device.

Args:
    device (torch.device or int or str, optional): selected device. Returns
        statistic for the current device, given by :f...

#### torch.xpu.seed
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Set the seed for generating random numbers to a random number for the current GPU.

It's safe to call this function if XPU is not available; in that case, it is silently ignored.

.. warning::
    If ...

#### torch.xpu.seed_all
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Set the seed for generating random numbers to a random number on all GPUs.

It's safe to call this function if XPU is not available; in that case, it is silently ignored....

#### torch.xpu.set_rng_state
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Set the random number generator state of the specified GPU.

Args:
    new_state (torch.ByteTensor): The desired state
    device (torch.device or int, optional): The device to set the RNG state.
    ...

#### torch.xpu.set_rng_state_all
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Set the random number generator state of all devices.

Args:
    new_states (Iterable of torch.ByteTensor): The desired state for each device....

#### torch.xpu.synchronize
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: Wait for all kernels in all streams on a XPU device to complete.

Args:
    device (torch.device or int, optional): device for which to synchronize.
        It uses the current device, given by :func:...

### 8. Memory Functions (Phase 3 - LOWER PRIORITY)

#### torch.OutOfMemoryError
- **Category**: TORCH_MEMORY
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Exception raised when device is out of memory...

#### torch.memory_format
- **Category**: TORCH_MEMORY
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: ...

#### torch._prims_common.check_pin_memory
- **Category**: CORE_TORCH
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: ...

#### torch._prims_common.is_contiguous_for_memory_format
- **Category**: CORE_TORCH
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: ...

#### torch._prims_common.suggest_memory_format
- **Category**: CORE_TORCH
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: ...

#### torch._prims_common.validate_memory_format
- **Category**: CORE_TORCH
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: ...

#### torch.mps.current_allocated_memory
- **Category**: MPS_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Returns the current GPU memory occupied by tensors in bytes.

.. note::
   The returned size does not include cached allocations in
   memory pools of MPSAllocator....

#### torch.mps.driver_allocated_memory
- **Category**: MPS_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Returns total GPU memory allocated by Metal driver for the process in bytes.

.. note::
   The returned size includes cached allocations in MPSAllocator pools
   as well as allocations from MPS/MPSGra...

#### torch.mps.recommended_max_memory
- **Category**: MPS_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Returns recommended max Working set size for GPU memory in bytes.

.. note::
   Recommended max working set size for Metal.
   returned from device.recommendedMaxWorkingSetSize....

#### torch.mps.set_per_process_memory_fraction
- **Category**: MPS_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Set memory fraction for limiting process's memory allocation on MPS device.
The allowed value equals the fraction multiplied by recommended maximum device memory
(obtained from Metal API device.recomm...

#### torch.mtia.max_memory_allocated
- **Category**: MTIA_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Return the maximum memory allocated in bytes for a given device.

Args:
    device (torch.device, str, or int, optional) selected device. Returns
        statistics for the current device, given by cu...

#### torch.mtia.memory_stats
- **Category**: MTIA_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Return a dictionary of MTIA memory allocator statistics for a given device.

Args:
    device (torch.device, str, or int, optional) selected device. Returns
        statistics for the current device, ...

#### torch.mtia.record_memory_history
- **Category**: MTIA_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Enable/Disable the memory profiler on MTIA allocator

Args:
    enabled (all or state, optional) selected device. Returns
        statistics for the current device, given by current_device(),
        ...

#### torch.mtia.reset_peak_memory_stats
- **Category**: MTIA_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Reset the peak memory stats for a given device.


Args:
    device (torch.device, str, or int, optional) selected device. Returns
        statistics for the current device, given by current_device(),
...

#### torch.xpu.max_memory_allocated
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Return the maximum GPU memory occupied by tensors in bytes for a given device.

By default, this returns the peak allocated memory since the beginning of
this program. :func:`~torch.xpu.reset_peak_mem...

#### torch.xpu.max_memory_reserved
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Return the maximum GPU memory managed by the caching allocator in bytes for a given device.

By default, this returns the peak cached memory since the beginning of this
program. :func:`~torch.xpu.rese...

#### torch.xpu.memory_allocated
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Return the current GPU memory occupied by tensors in bytes for a given device.

Args:
    device (torch.device or int or str, optional): selected device. Returns
        statistic for the current devi...

#### torch.xpu.memory_reserved
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Return the current GPU memory managed by the caching allocator in bytes for a given device.

Args:
    device (torch.device or int or str, optional): selected device. Returns
        statistic for the...

#### torch.xpu.memory_stats
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Return a dictionary of XPU memory allocator statistics for a given device.

The return value of this function is a dictionary of statistics, each of
which is a non-negative integer.

Core statistics:
...

#### torch.xpu.memory_stats_as_nested_dict
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Return the result of :func:`~torch.xpu.memory_stats` as a nested dictionary....

#### torch.xpu.reset_accumulated_memory_stats
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Reset the "accumulated" (historical) stats tracked by the XPU memory allocator.

See :func:`~torch.xpu.memory_stats` for details. Accumulated stats correspond to
the `"allocated"` and `"freed"` keys i...

#### torch.xpu.reset_peak_memory_stats
- **Category**: XPU_OPERATIONS
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: Reset the "peak" stats tracked by the XPU memory allocator.

See :func:`~torch.xpu.memory_stats` for details. Peak stats correspond to the
`"peak"` key in each individual stat dict.

Args:
    device ...

## Implementation Phases Summary

### Phase 1: Core Infrastructure (Highest Priority)
- Device creation functions
- Tensor creation functions
- **Total**: 253 functions

### Phase 2: Device Management (Medium Priority)
- Device management functions
- Events and streams
- **Total**: 31 functions

### Phase 3: Advanced Features (Lower Priority)
- Neural network functions
- Device-specific operations
- Memory management
- **Total**: 1241 functions

## Next Steps

1. **Start with Phase 1** - Implement core device and tensor creation
2. **Move to Phase 2** - Add device management and CUDA compatibility
3. **Complete Phase 3** - Handle advanced features and optimizations
4. **Test thoroughly** - Ensure all functions work correctly
5. **Update status** - Mark functions as implemented in migration plan
