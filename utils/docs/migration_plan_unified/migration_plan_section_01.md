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

| 🟦 CORE_TORCH | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
|:-----------------|:---:|:----:|:---:|:---:|:------:|-----------|-------------|-------|
| `torch.align_tensors` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` |  |
| `torch.are_deterministic_algorithms_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Returns True if the global deterministic flag is turned on. Refer to :func:`torch.use_deterministic_algorithms` documentation for more details. |
| `torch.atleast_1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` | Returns a 1-dimensional view of each input tensor with zero dimensions. Input tensors with one or more dimensions are returned as-is. Args: input (Tensor or list of Tensors) Returns: output (Tensor... |
| `torch.atleast_2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` | Returns a 2-dimensional view of each input tensor with zero dimensions. Input tensors with two or more dimensions are returned as-is. Args: input (Tensor or list of Tensors) Returns: output (Tensor... |
| `torch.atleast_3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` | Returns a 3-dimensional view of each input tensor with zero dimensions. Input tensors with three or more dimensions are returned as-is. Args: input (Tensor or list of Tensors) Returns: output (Tens... |
| `torch.block_diag` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` | Create a block diagonal matrix from provided tensors. Args: *tensors: One or more tensors with 0, 1, or 2 dimensions. Returns: Tensor: A 2 dimensional tensor with all the input tensors arranged in ... |
| `torch.broadcast_shapes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shapes` | `Any` | broadcast_shapes(*shapes) -> Size Similar to :func:`broadcast_tensors` but for shapes. This is equivalent to ``torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape`` but avoids the need crea... |
| `torch.broadcast_tensors` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` | broadcast_tensors(*tensors) -> List of Tensors Broadcasts the given tensors according to :ref:`broadcasting-semantics`. Args: *tensors: any number of tensors of the same type .. warning:: More than... |
| `torch.cartesian_prod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `<class 'torch.Tensor'>` | Do cartesian product of the given sequence of tensors. The behavior is similar to python's `itertools.product`. Args: *tensors: any number of 1 dimensional tensors. Returns: Tensor: A tensor equiva... |
| `torch.cdist` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, p, ...` | `Any` | Computes batched the p-norm distance between each pair of the two collections of row vectors. Args: x1 (Tensor): input tensor where the last two dimensions represent the points and the feature dime... |
| `torch.chain_matmul` | ❓ | ❓ | ❓ | ❓ | 🔴 | `matrices, out` | `Any` | Returns the matrix product of the :math:`N` 2-D tensors. This product is efficiently computed using the matrix chain order algorithm which selects the order in which incurs the lowest cost in terms... |
| `torch.classproperty` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` |  |
| `torch.compile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, fullgraph, dynamic, ...` | `typing.Union[typing.Callable[[typing.Callable[~_InputT, ~_RetT]], typing.Callable[~_InputT, ~_RetT]], typing.Callable[~_InputT, ~_RetT]]` | Optimizes given model/function using TorchDynamo and specified backend. If you are compiling an :class:`torch.nn.Module`, you can also use :meth:`torch.nn.Module.compile` to compile the module inpl... |
| `torch.compiled_with_cxx11_abi` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1 |
| `torch.cond` | ❓ | ❓ | ❓ | ❓ | 🔴 | `pred, true_fn, false_fn, ...` | `typing.Any` | Conditionally applies `true_fn` or `false_fn`. .. warning:: `torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types and doesn't support training currently.... |
| `torch.eig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, eigenvectors, e, ...` | `tuple[torch.Tensor, torch.Tensor]` |  |
| `torch.einsum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args` | `<class 'torch.Tensor'>` | einsum(equation, *operands) -> Tensor Sums the product of the elements of the input :attr:`operands` along dimensions specified using a notation based on the Einstein summation convention. Einsum a... |
| `torch.from_dlpack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ext_tensor` | `torch.Tensor` | from_dlpack(ext_tensor) -> Tensor Converts a tensor from an external library into a ``torch.Tensor``. The returned PyTorch tensor will share the memory with the input tensor (which may have come fr... |
| `torch.get_default_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `torch.device` | Gets the default ``torch.Tensor`` to be allocated on ``device`` |
| `torch.get_deterministic_debug_mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Returns the current value of the debug mode for deterministic operations. Refer to :func:`torch.set_deterministic_debug_mode` documentation for more details. |
| `torch.get_file_path` | ❓ | ❓ | ❓ | ❓ | 🔴 | `path_components` | `<class 'str'>` |  |
| `torch.get_float32_matmul_precision` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'str'>` | Returns the current value of float32 matrix multiplication precision. Refer to :func:`torch.set_float32_matmul_precision` documentation for more details. |
| `torch.get_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'torch.Tensor'>` | Returns the random number generator state as a `torch.ByteTensor`. .. note:: The returned state is for the default generator on CPU only. See also: :func:`torch.random.fork_rng`. |
| `torch.initial_seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Returns the initial seed for generating random numbers as a Python `long`. .. note:: The returned seed is for the default generator on CPU only. |
| `torch.is_deterministic_algorithms_warn_only_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Returns True if the global deterministic flag is set to warn only. Refer to :func:`torch.use_deterministic_algorithms` documentation for more details. |
| `torch.is_storage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj` | `typing_extensions.TypeIs[typing.Union[ForwardRef('TypedStorage'), ForwardRef('UntypedStorage')]]` | Returns True if `obj` is a PyTorch storage object. Args: obj (Object): Object to test |
| `torch.is_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj` | `typing_extensions.TypeIs[ForwardRef('torch.Tensor')]` | Returns True if `obj` is a PyTorch tensor. Note that this function is simply doing ``isinstance(obj, Tensor)``. Using that ``isinstance`` check is better for typechecking with mypy, and more explic... |
| `torch.is_warn_always_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Returns True if the global warn_always flag is turned on. Refer to :func:`torch.set_warn_always` documentation for more details. |
| `torch.load` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f, map_location, pickle_module, ...` | `typing.Any` | load(f, map_location=None, pickle_module=pickle, *, weights_only=True, mmap=None, **pickle_load_args) Loads an object saved with :func:`torch.save` from a file. :func:`torch.load` uses Python's unp... |
| `torch.lobpcg` | ❓ | ❓ | ❓ | ❓ | 🔴 | `A, k, B, ...` | `tuple[torch.Tensor, torch.Tensor]` | Find the k largest (or smallest) eigenvalues and the corresponding eigenvectors of a symmetric positive definite generalized eigenvalue problem using matrix-free LOBPCG methods. This function is a ... |
| `torch.lstsq` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, A, out` | `tuple[torch.Tensor, torch.Tensor]` |  |
| `torch.lu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Computes the LU factorization of a matrix or batches of matrices :attr:`A`. Returns a tuple containing the LU factorization and pivots of :attr:`A`. Pivoting is done if :attr:`pivot` is set to ``Tr... |
| `torch.manual_seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `seed` | `<class 'torch._C.Generator'>` | Sets the seed for generating random numbers on all devices. Returns a `torch.Generator` object. Args: seed (int): The desired seed. Value must be within the inclusive range `[-0x8000_0000_0000_0000... |
| `torch.matrix_rank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, tol, symmetric, ...` | `<class 'torch.Tensor'>` |  |
| `torch.meshgrid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, indexing` | `tuple[torch.Tensor, ...]` | Creates grids of coordinates specified by the 1D inputs in `attr`:tensors. This is helpful when you want to visualize data over some range of inputs. See below for a plotting example. Given :math:`... |
| `torch.norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, p, dim, ...` | `Any` | Returns the matrix norm or vector norm of a given tensor. .. warning:: torch.norm is deprecated and may be removed in a future PyTorch release. Its documentation and behavior may be incorrect, and ... |
| `torch.pca_lowrank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `A, q, center, ...` | `tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | Performs linear Principal Component Analysis (PCA) on a low-rank matrix, batches of such matrices, or sparse matrix. This function returns a namedtuple ``(U, S, V)`` which is the nearly optimal app... |
| `torch.prepare_multiprocessing_environment` | ❓ | ❓ | ❓ | ❓ | 🔴 | `path` | `None` |  |
| `torch.profiler_allow_cudagraph_cupti_lazy_reinit_cuda12` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.save` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj, f, pickle_module, ...` | `None` | save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True) Saves an object to a disk file. See also: :ref:`saving-loading-tensors` Args: obj: saved object f: a file-... |
| `torch.seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Sets the seed for generating random numbers to a non-deterministic random number on all devices. Returns a 64 bit number used to seed the RNG. |
| `torch.set_default_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Sets the default ``torch.Tensor`` to be allocated on ``device``. This does not affect factory function calls which are called with an explicit ``device`` argument. Factory calls will be performed a... |
| `torch.set_default_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `d` | `None` | Sets the default floating point dtype to :attr:`d`. Supports floating point dtype as inputs. Other dtypes will cause torch to raise an exception. When PyTorch is initialized its default floating po... |
| `torch.set_default_tensor_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `t` | `None` | .. warning:: This function is deprecated as of PyTorch 2.1, please use :func:`torch.set_default_dtype()` and :func:`torch.set_default_device()` as alternatives. Sets the default ``torch.Tensor`` ty... |
| `torch.set_deterministic_debug_mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `debug_mode` | `None` | Sets the debug mode for deterministic operations. .. note:: This is an alternative interface for :func:`torch.use_deterministic_algorithms`. Refer to that function's documentation for details about... |
| `torch.set_float32_matmul_precision` | ❓ | ❓ | ❓ | ❓ | 🔴 | `precision` | `None` | Sets the internal precision of float32 matrix multiplications. Running float32 matrix multiplications in lower precision may significantly increase performance, and in some programs the loss of pre... |
| `torch.set_printoptions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `precision, threshold, edgeitems, ...` | `Any` | Set options for printing. Items shamelessly taken from NumPy Args: precision: Number of digits of precision for floating point output (default = 4). threshold: Total number of array elements which ... |
| `torch.set_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `new_state` | `None` | Sets the random number generator state. .. note:: This function only works for CPU. For CUDA, please use :func:`torch.manual_seed`, which works for both CPU and CUDA. Args: new_state (torch.ByteTen... |
| `torch.set_warn_always` | ❓ | ❓ | ❓ | ❓ | 🔴 | `b` | `None` | When this flag is False (default) then some PyTorch warnings may only appear once per process. This helps avoid excessive warning information. Setting it to True causes these warnings to always app... |
| `torch.solve` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, A, out` | `tuple[torch.Tensor, torch.Tensor]` |  |
| `torch.split` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, split_size_or_sections, dim` | `tuple[torch.Tensor, ...]` | Splits the tensor into chunks. Each chunk is a view of the original tensor. If :attr:`split_size_or_sections` is an integer type, then :attr:`tensor` will be split into equally sized chunks (if pos... |
| `torch.stft` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, n_fft, hop_length, ...` | `<class 'torch.Tensor'>` | Short-time Fourier transform (STFT). .. warning:: From version 1.8.0, :attr:`return_complex` must always be given explicitly for real inputs and `return_complex=False` has been deprecated. Strongly... |
| `torch.svd_lowrank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `A, q, niter, ...` | `tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | Return the singular value decomposition ``(U, S, V)`` of a matrix, batches of matrices, or a sparse matrix :math:`A` such that :math:`A \approx U \operatorname{diag}(S) V^{\text{H}}`. In case :math... |
| `torch.sym_float` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` | SymInt-aware utility for float casting. Args: a (SymInt, SymFloat, or object): Object to cast |
| `torch.sym_fresh_size` | ❓ | ❓ | ❓ | ❓ | 🔴 | `expr` | `Any` |  |
| `torch.sym_int` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` | SymInt-aware utility for int casting. Args: a (SymInt, SymFloat, or object): Object to cast |
| `torch.sym_ite` | ❓ | ❓ | ❓ | ❓ | 🔴 | `b, t, f` | `Any` |  |
| `torch.sym_max` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `Any` | SymInt-aware utility for max which avoids branching on a < b. Unlike builtins.max(), this only works for int/float, and it always promotes to float if any argument is float (unlike builtins.max, wh... |
| `torch.sym_min` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `Any` | SymInt-aware utility for min(). |
| `torch.sym_not` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` | SymInt-aware utility for logical negation. Args: a (SymBool or bool): Object to negate |
| `torch.sym_sqrt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch.sym_sum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args` | `Any` | N-ary add which is faster to compute for long lists than iterated binary addition. Only does something special for integers. |
| `torch.symeig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, eigenvectors, upper, ...` | `tuple[torch.Tensor, torch.Tensor]` |  |
| `torch.tensordot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, dims, ...` | `Any` | Returns a contraction of a and b over multiple dimensions. :attr:`tensordot` implements a generalized matrix product. Args: a (Tensor): Left tensor to contract b (Tensor): Right tensor to contract ... |
| `torch.typename` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj` | `<class 'str'>` | String representation of the type of an object. This function returns a fully qualified string representation of an object's type. Args: obj (object): The object whose type to represent Returns: st... |
| `torch.unique` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> tuple[Tensor, Tensor, Tensor] Returns the unique elements of the input tensor. .. note:: This function is differen... |
| `torch.unique_consecutive` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Eliminates all but the first element from every consecutive group of equivalent elements. .. note:: This function is different from :func:`torch.unique` in the sense that this function only elimina... |
| `torch.unravel_index` | ❓ | ❓ | ❓ | ❓ | 🔴 | `indices, shape` | `tuple[torch.Tensor, ...]` | Converts a tensor of flat indices into a tuple of coordinate tensors that index into an arbitrary tensor of the specified shape. Args: indices (Tensor): An integer tensor containing indices into th... |
| `torch.use_deterministic_algorithms` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mode, warn_only` | `None` | Sets whether PyTorch operations must use "deterministic" algorithms. That is, algorithms which, given the same input, and when run on the same software and hardware, always produce the same output.... |
| `torch.vmap` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, in_dims, out_dims, ...` | `typing.Callable` | vmap is the vectorizing map; ``vmap(func)`` returns a new function that maps ``func`` over some dimension of the inputs. Semantically, vmap pushes the map into PyTorch operations called by ``func``... |
| `torch.while_loop` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cond_fn, body_fn, carried_inputs` | `Any` | Run body_fn(*carried_inputs) while cond_fn(*carried_inputs) returns a True scalar tensor. Returns the output of body_fn or initial carried_inputs. .. warning:: `torch.while_loop` is a prototype fea... |
| `torch._decomp.core_aten_decompositions` | ❓ | ❓ | ❓ | ❓ | ✅ | `` | `CustomDecompTable` |  |
| `torch._decomp.get_decompositions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `aten_ops, type` | `dict[torch._ops.OperatorBase, typing.Callable]` | Retrieve a dictionary of decompositions corresponding to the list of operator overloads and overload packets passed as input. Overload packets will include all decomposed overloads in the packet. I... |
| `torch._decomp.lru_cache` | ❓ | ❓ | ❓ | ❓ | 🔴 | `maxsize, typed` | `Any` | Least-recently-used cache decorator. If *maxsize* is set to None, the LRU features are disabled and the cache can grow without bound. If *typed* is True, arguments of different types will be cached... |
| `torch._decomp.register_decomposition` | ❓ | ❓ | ❓ | ❓ | 🔴 | `aten_op, registry, type, ...` | `typing.Callable[[typing.Callable[~_P, ~_T]], typing.Callable[~_P, ~_T]]` | A decorator to register a function as a decomposition to the Python decomposition table. Use it like this:: @register_decomposition(torch.ops.aten.clamp_min) def clamp_min(x): return torch.clamp(se... |
| `torch._decomp.remove_decompositions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `decompositions, aten_ops` | `None` | Given a dictionary of decompositions obtained from get_decompositions(), removes operators associated with a list of operator overloads and overload packets passed as input. If the decomposition di... |
| `torch._decomp.wraps` | ❓ | ❓ | ❓ | ❓ | 🔴 | `wrapped, assigned, updated` | `Any` | Decorator factory to apply update_wrapper() to a wrapper function Returns a decorator that invokes update_wrapper() with the decorated function as the wrapper argument and the arguments to wraps() ... |
| `torch._dynamo.allow_in_graph` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` | Tells the compiler frontend (Dynamo) to skip symbolic introspection of the function and instead directly write it to the graph when encountered. See :func:`torch.compiler.allow_in_graph`'s docstrin... |
| `torch._dynamo.assume_constant_result` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` |  |
| `torch._dynamo.disable` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn, recursive, reason` | `Any` | Decorator to disable TorchDynamo If recursive=True, Dynamo is completely skipped on the decorated function frame as well as the recursively invoked functions. If recursive=False, Dynamo skips frame... |
| `torch._dynamo.disallow_in_graph` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` | Customize which functions TorchDynamo will exclude in the generated graph and force a graph break on. :: torch._dynamo.disallow_in_graph(torch.sub) @torch._dynamo.optimize(...) def fn(a): x = torch... |
| `torch._dynamo.explain` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f, extra_args, extra_kwargs` | `Any` |  |
| `torch._dynamo.export` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f, extra_args, aten_graph, ...` | `Callable[..., ExportResult]` | Export an input function f to a format that can be executed outside of PyTorch using the FX graph. Args: f (callable): A PyTorch function to be exported. aten_graph (bool): If True, exports a graph... |
| `torch._dynamo.forbid_in_graph` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` | Customize which functions TorchDynamo will assert are not present while tracing. If you want a graph break on this function instead, use disallow_in_graph. TODO(voz): We now have allow_in_graph, di... |
| `torch._dynamo.graph_break` | ❓ | ❓ | ❓ | ❓ | 🔴 | `msg` | `Any` | Force a graph break |
| `torch._dynamo.is_compiling` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Indicates whether we are tracing/compiling with torch.compile() or torch.export(). |
| `torch._dynamo.is_dynamo_supported` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._dynamo.is_inductor_supported` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._dynamo.list_backends` | ❓ | ❓ | ❓ | ❓ | 🔴 | `exclude_tags` | `list[str]` | Return valid strings that can be passed to: torch.compile(..., backend="name") |
| `torch._dynamo.lookup_backend` | ❓ | ❓ | ❓ | ❓ | 🔴 | `compiler_fn` | `Any` | Expand backend strings to functions |
| `torch._dynamo.mark_dynamic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `t, index, min, ...` | `Any` | Mark a tensor as having a dynamic dim and set corresponding min and max range for the dim. [Note - on the state of mark_dynamic] The behavior of having a dynamic dimension on a tensor is governed b... |
| `torch._dynamo.mark_static` | ❓ | ❓ | ❓ | ❓ | 🔴 | `t, index` | `Any` | Mark a tensor as having a static dim or mark a nn module class as static. For tensors =========== This will prevent us from attempting to compile it dynamically when dynamic=True; this can improve ... |
| `torch._dynamo.mark_static_address` | ❓ | ❓ | ❓ | ❓ | 🔴 | `t, guard` | `Any` | Marks an input tensor whose data_ptr will not change across multiple calls to a dynamo-compiled function. This indicates to cudagraphs that an extra allocation is not needed for this input. The dat... |
| `torch._dynamo.maybe_mark_dynamic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `t, index` | `Any` | Mark a tensor as having a dynamic dim, but don't enforce it (i.e., if this dimension ends up getting specialized, don't error). |
| `torch._dynamo.nonstrict_trace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `traceable_fn` | `Any` |  |
| `torch._dynamo.on_compile_end` | ❓ | ❓ | ❓ | ❓ | 🔴 | `callback` | `typing.Callable[[], NoneType]` | Decorator to register a callback function for the end of the compilation. |
| `torch._dynamo.on_compile_start` | ❓ | ❓ | ❓ | ❓ | 🔴 | `callback` | `typing.Callable[[], NoneType]` | Decorator to register a callback function for the start of the compilation. |
| `torch._dynamo.optimize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` |  |
| `torch._dynamo.optimize_assert` | ❓ | ❓ | ❓ | ❓ | 🔴 | `backend, hooks, export, ...` | `Any` | The same as `torch._dynamo.optimize(backend, nopython=True)` |
| `torch._dynamo.register_backend` | ❓ | ❓ | ❓ | ❓ | 🔴 | `compiler_fn, name, tags` | `Any` | Decorator to add a given compiler to the registry to allow calling `torch.compile` with string shorthand. Note: for projects not imported by default, it might be easier to pass a function directly ... |
| `torch._dynamo.replay` | ❓ | ❓ | ❓ | ❓ | 🔴 | `filename` | `None` |  |
| `torch._dynamo.reset` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Clear all compile caches and restore initial state. This function is intended to reset Dynamo's state *as if* you had started a fresh process invocation, which makes it good for testing scenarios w... |
| `torch._dynamo.reset_code_caches` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Clears in-memory code cache, which is what stores compiled products. This resets less state than :func:`reset` and is mostly only used for testing purposes. |
| `torch._dynamo.reset_code_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` |  |
| `torch._dynamo.reset_frame_count` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` |  |
| `torch._dynamo.run` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` | Don't do any dynamic compiles, just use prior optimizations |
| `torch._dynamo.substitute_in_graph` | ❓ | ❓ | ❓ | ❓ | 🔴 | `original_fn, can_constant_fold_through, skip_signature_check, ...` | `typing.Callable[[typing.Callable[~_P, ~_R]], typing.Callable[~_P, ~_R]]` | Register a polyfill handler for a function, usually a C function from the C extension, to be used in place of the original function when inlining the original function in the graph. .. note:: The p... |
| `torch._export.aot_compile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f, args, kwargs, ...` | `typing.Union[list[str], str]` | Note: this function is not stable yet Traces either an nn.Module's forward function or just a callable with PyTorch operations inside, generates executable cpp code from the program, and returns th... |
| `torch._export.aot_load` | ❓ | ❓ | ❓ | ❓ | 🔴 | `so_path, device` | `typing.Callable` | Loads a shared library generated by aot_compile and returns a callable Args: so_path: Path to the shared library Returns: A callable |
| `torch._export.compatibility` | ❓ | ❓ | ❓ | ❓ | 🔴 | `is_backward_compatible` | `typing.Callable[[~_T], ~_T]` |  |
| `torch._export.compile_context` | ❓ | ❓ | ❓ | ❓ | 🔴 | `context` | `Any` |  |
| `torch._export.contextmanager` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` | @contextmanager decorator. Typical usage: @contextmanager def some_generator(<arguments>): <setup> try: yield <value> finally: <cleanup> This makes this: with some_generator(<arguments>) as <variab... |
| `torch._export.log_export_usage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kwargs` | `Any` |  |
| `torch._export.lru_cache` | ❓ | ❓ | ❓ | ❓ | 🔴 | `maxsize, typed` | `Any` | Least-recently-used cache decorator. If *maxsize* is set to None, the LRU features are disabled and the cache can grow without bound. If *typed* is True, arguments of different types will be cached... |
| `torch._export.make_fx` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f, decomposition_table, tracing_mode, ...` | `Callable[..., GraphModule]` | Given a function f, return a new function which when executed with valid arguments to f, returns an FX GraphModule representing the set of operations that were executed during the course of execution. |
| `torch._export.patch` | ❓ | ❓ | ❓ | ❓ | 🔴 | `target, new, spec, ...` | `Any` | `patch` acts as a function decorator, class decorator or a context manager. Inside the body of the function or with statement, the `target` is patched with a `new` object. When the function/with st... |
| `torch._export.reorder_kwargs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `user_kwargs, spec` | `dict[str, typing.Any]` | Reorder user-provided kwargs to match the order in `spec`. `spec` is expected to be the in_spec of an exported program, i.e. the spec that results from flattening `(args, kwargs)`. We need this to ... |
| `torch._higher_order_ops.associative_scan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `combine_fn, xs, dim, ...` | `<class 'torch.Tensor'>` | Performs an inclusive scan with an associative combine function. .. warning:: `torch.associative_scan` is a prototype feature in PyTorch. It currently does not support autograd and you may run into... |
| `torch._higher_order_ops.cond` | ❓ | ❓ | ❓ | ❓ | 🔴 | `pred, true_fn, false_fn, ...` | `typing.Any` | Conditionally applies `true_fn` or `false_fn`. .. warning:: `torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types and doesn't support training currently.... |
| `torch._higher_order_ops.foreach_map` | ❓ | ❓ | ❓ | ❓ | 🔴 | `op, operands, kwargs` | `Any` |  |
| `torch._higher_order_ops.scan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `combine_fn, init, xs, ...` | `tuple[typing.Any, typing.Any]` | Performs an inclusive scan with a combine function. .. warning:: `torch.scan` is a prototype feature in PyTorch. It currently does not support autograd and you may run into miscompiles. Read more a... |
| `torch._higher_order_ops.strict_mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `callable, operands` | `Any` |  |
| `torch._higher_order_ops.while_loop` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cond_fn, body_fn, carried_inputs` | `Any` | Run body_fn(*carried_inputs) while cond_fn(*carried_inputs) returns a True scalar tensor. Returns the output of body_fn or initial carried_inputs. .. warning:: `torch.while_loop` is a prototype fea... |
| `torch._inductor.aot_compile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `gm, args, kwargs, ...` | `Union[str, list[str]]` | Ahead-of-time compile a given FX graph with TorchInductor into a shared library. Args: gm: The FX graph to compile. args: Example arguments kwargs: Example keyword arguments options: Optional dict ... |
| `torch._inductor.aoti_compile_and_package` | ❓ | ❓ | ❓ | ❓ | 🔴 | `exported_program, _deprecated_unused_args, _deprecated_unused_kwargs, ...` | `str` | Compiles the exported program with AOTInductor, and packages it into a .pt2 artifact specified by the input package_path. To load the package, you can call ``torch._inductor.aoti_load_package(packa... |
| `torch._inductor.aoti_load_package` | ❓ | ❓ | ❓ | ❓ | 🔴 | `path, run_single_threaded` | `Any` | Loads the model from the PT2 package. If multiple models were packaged into the PT2, this will load the default model. To load a specific model, you can directly call the load API .. code-block:: p... |
| `torch._inductor.compile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `gm, example_inputs, options` | `Any` | Compile a given FX graph with TorchInductor. This allows compiling FX graphs captured without using TorchDynamo. Args: gm: The FX graph to compile. example_inputs: List of tensor inputs. options: O... |
| `torch._inductor.cudagraph_mark_step_begin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Indicates that a new iteration of inference or training is about to begin. |
| `torch._inductor.list_mode_options` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mode, dynamic` | `dict[str, Any]` | Returns a dictionary describing the optimizations that each of the available modes passed to `torch.compile()` performs. Args: mode (str, optional): The mode to return the optimizations for. If Non... |
| `torch._inductor.list_options` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `list[str]` | Returns a dictionary describing the optimizations and debug configurations that are available to `torch.compile()`. The options are documented in `torch._inductor.config`. Example:: >>> torch._indu... |
| `torch._inductor.standalone_compile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `gm, example_inputs, options` | `CompiledArtifact` | Precompilation API for inductor. .. code-block:: python compiled_artifact = torch._inductor.standalone_compile(gm, args) compiled_artifact.save(path=path, format="binary") # Later on a new process ... |
| `torch._lazy.add_step_closure` | ❓ | ❓ | ❓ | ❓ | 🔴 | `closure, args, run_async` | `Any` | Adds a closure to the list of the ones to be run at the end of the step. Many times during model training there is the need to print/report (print to console, post to tensorboard, etc...) informati... |
| `torch._lazy.get_tensor_id` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor` | `Any` | Return a unique id of the lazy tensor maintained by LTC |
| `torch._lazy.mark_step` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device, wait` | `Any` | Triggers a mark step, which amounts to - collecting a group of 'live' lazy tensors to index into the compilation cache (lowering/compiling their IR graphs if not cached) - kicking off execution of ... |
| `torch._lazy.run_step_closures` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._lazy.save` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, args, kwargs` | `Any` |  |
| `torch._lazy.sync_multi` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, devices` | `Any` | Sync the list of lazy tensors so there IR get lowered for the activate backend and the compiled computation graph get cached. |
| `torch._lazy.to_cpu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, devices` | `Any` |  |
| `torch._lazy.tree_flatten` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tree, is_leaf` | `tuple[list[typing.Any], torch.utils._pytree.TreeSpec]` | Flattens a pytree into a list of values and a TreeSpec that can be used to reconstruct the pytree. |
| `torch._lazy.tree_unflatten` | ❓ | ❓ | ❓ | ❓ | 🔴 | `leaves, treespec` | `typing.Any` | Given a list of values and a TreeSpec, builds a pytree. This is the inverse operation of `tree_flatten`. |
| `torch._lazy.wait_device_ops` | ❓ | ❓ | ❓ | ❓ | 🔴 | `devices` | `Any` | Waits for all the async operations on the given devices to complete. Args: devices (string..., optional): The devices whose async ops need to be waited for. If empty, all the local devices will be ... |
| `torch._library.capture_triton` | ❓ | ❓ | ❓ | ❓ | 🔴 | `triton_kernel` | `typing.Any` | This API has been renamed to wrap_triton |
| `torch._library.register_fake_class` | ❓ | ❓ | ❓ | ❓ | 🔴 | `qualname, fake_class` | `Any` | Register a fake implementation for this class. It's in the same spirit of registering a fake implementation for an operator but with the difference that it associates a fake class with the original... |
| `torch._library.triton_op` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, fn, mutates_args, ...` | `typing.Callable` | Create a custom operator whose implementation is backed by 1+ triton kernels. This is a more structured way of using triton kernels with PyTorch. Prefer using triton kernels with no ``torch.library... |
| `torch._library.wrap_triton` | ❓ | ❓ | ❓ | ❓ | 🔴 | `triton_kernel` | `typing.Any` | Allows capture of a triton kernel into a graph via make_fx or non-strict ``torch.export``. These technologies perform Dispatcher-based tracing (via ``__torch_dispatch__``) and cannot see calls to r... |
| `torch._logging.dtrace_structured` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, metadata_fn, payload_fn, ...` | `Any` | For logging more detailed information used for debugging. This may result in the program becoming slow. |
| `torch._logging.getArtifactLogger` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module_qname, artifact_name` | `Any` |  |
| `torch._logging.get_structured_logging_overhead` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `typing.Optional[float]` |  |
| `torch._logging.set_logs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `all, dynamo, aot, ...` | `Any` | Sets the log level for individual components and toggles individual log artifact types. .. warning:: This feature is a prototype and may have compatibility breaking changes in the future. .. note::... |
| `torch._logging.trace_structured` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, metadata_fn, payload_fn, ...` | `None` | metadata is an arbitrary JSON compatible struct, but it's expected to not be too long (e.g., less than 1MB) payload is an arbitrary string, which can be arbitrarily long (but expected to have newli... |
| `torch._numpy.abs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.absolute` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.add` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, out, ...` | `Any` |  |
| `torch._numpy.allclose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, rtol, ...` | `Any` |  |
| `torch._numpy.alltrue` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, out, ...` | `Any` |  |
| `torch._numpy.amax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, out, ...` | `Any` |  |
| `torch._numpy.amin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, out, ...` | `Any` |  |
| `torch._numpy.angle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `z, deg` | `Any` |  |
| `torch._numpy.any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, out, ...` | `Any` |  |
| `torch._numpy.append` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arr, values, axis` | `Any` |  |
| `torch._numpy.arange` | ❓ | ❓ | ❓ | ❓ | 🔴 | `start, stop, step, ...` | `Any` |  |
| `torch._numpy.arccos` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.arccosh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.arcsin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.arcsinh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.arctan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.arctan2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.arctanh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.argmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, out, ...` | `Any` |  |
| `torch._numpy.argmin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, out, ...` | `Any` |  |
| `torch._numpy.argsort` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, kind, ...` | `Any` |  |
| `torch._numpy.argwhere` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch._numpy.around` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, decimals, out` | `Any` |  |
| `torch._numpy.array` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj, dtype, copy, ...` | `Any` |  |
| `torch._numpy.array_equal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a1, a2, equal_nan` | `Any` |  |
| `torch._numpy.array_equiv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a1, a2` | `Any` |  |
| `torch._numpy.array_split` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ary, indices_or_sections, axis` | `Any` |  |
| `torch._numpy.asarray` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dtype, order, ...` | `Any` |  |
| `torch._numpy.ascontiguousarray` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dtype, like` | `Any` |  |
| `torch._numpy.atleast_1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arys` | `Any` |  |
| `torch._numpy.atleast_2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arys` | `Any` |  |
| `torch._numpy.atleast_3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arys` | `Any` |  |
| `torch._numpy.average` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, weights, ...` | `Any` |  |
| `torch._numpy.bartlett` | ❓ | ❓ | ❓ | ❓ | 🔴 | `M` | `Any` |  |
| `torch._numpy.bincount` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, weights, minlength` | `Any` |  |
| `torch._numpy.bitwise_and` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.bitwise_not` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.bitwise_or` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.bitwise_xor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.blackman` | ❓ | ❓ | ❓ | ❓ | 🔴 | `M` | `Any` |  |
| `torch._numpy.broadcast_arrays` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, subok` | `Any` |  |
| `torch._numpy.broadcast_shapes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shapes` | `Any` | broadcast_shapes(*shapes) -> Size Similar to :func:`broadcast_tensors` but for shapes. This is equivalent to ``torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape`` but avoids the need crea... |
| `torch._numpy.broadcast_to` | ❓ | ❓ | ❓ | ❓ | 🔴 | `array, shape, subok` | `Any` |  |
| `torch._numpy.can_cast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `from_, to, casting` | `Any` |  |
| `torch._numpy.cbrt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.ceil` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.choose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, choices, out, ...` | `Any` |  |
| `torch._numpy.clip` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, min, max, ...` | `Any` |  |
| `torch._numpy.column_stack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tup, dtype, casting` | `Any` |  |
| `torch._numpy.common_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` |  |
| `torch._numpy.concatenate` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ar_tuple, axis, out, ...` | `Any` |  |
| `torch._numpy.conj` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.conjugate` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.convolve` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, v, mode` | `Any` |  |
| `torch._numpy.copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, order, subok` | `Any` |  |
| `torch._numpy.copysign` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.copyto` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dst, src, casting, ...` | `Any` |  |
| `torch._numpy.corrcoef` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, y, rowvar, ...` | `Any` |  |
| `torch._numpy.correlate` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, v, mode` | `Any` |  |
| `torch._numpy.cos` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.cosh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.count_nonzero` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, keepdims` | `Any` |  |
| `torch._numpy.cov` | ❓ | ❓ | ❓ | ❓ | 🔴 | `m, y, rowvar, ...` | `Any` |  |
| `torch._numpy.cross` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, axisa, ...` | `Any` |  |
| `torch._numpy.cumprod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, dtype, ...` | `Any` |  |
| `torch._numpy.cumproduct` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, dtype, ...` | `Any` |  |
| `torch._numpy.cumsum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, dtype, ...` | `Any` |  |
| `torch._numpy.deg2rad` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.degrees` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.diag` | ❓ | ❓ | ❓ | ❓ | 🔴 | `v, k` | `Any` |  |
| `torch._numpy.diag_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `n, ndim` | `Any` |  |
| `torch._numpy.diag_indices_from` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arr` | `Any` |  |
| `torch._numpy.diagflat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `v, k` | `Any` |  |
| `torch._numpy.diagonal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, offset, axis1, ...` | `Any` |  |
| `torch._numpy.diff` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, n, axis, ...` | `Any` |  |
| `torch._numpy.divide` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.divmod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out1, ...` | `Any` |  |
| `torch._numpy.dot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `Any` |  |
| `torch._numpy.dsplit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ary, indices_or_sections` | `Any` |  |
| `torch._numpy.dstack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tup, dtype, casting` | `Any` |  |
| `torch._numpy.dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arg` | `Any` |  |
| `torch._numpy.einsum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `operands, out, dtype, ...` | `Any` |  |
| `torch._numpy.empty` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, dtype, order, ...` | `Any` |  |
| `torch._numpy.empty_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `prototype, dtype, order, ...` | `Any` |  |
| `torch._numpy.equal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.exp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.exp2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.expand_dims` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis` | `Any` |  |
| `torch._numpy.expm1` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.eye` | ❓ | ❓ | ❓ | ❓ | 🔴 | `N, M, k, ...` | `Any` |  |
| `torch._numpy.fabs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.fill_diagonal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, val, wrap` | `Any` |  |
| `torch._numpy.finfo` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtyp` | `Any` |  |
| `torch._numpy.fix` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.flatnonzero` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch._numpy.flip` | ❓ | ❓ | ❓ | ❓ | 🔴 | `m, axis` | `Any` |  |
| `torch._numpy.fliplr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `m` | `Any` |  |
| `torch._numpy.flipud` | ❓ | ❓ | ❓ | ❓ | 🔴 | `m` | `Any` |  |
| `torch._numpy.float_power` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.floor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.floor_divide` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.fmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.fmin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.fmod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.from_dlpack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `Any` |  |
| `torch._numpy.full` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, fill_value, dtype, ...` | `Any` |  |
| `torch._numpy.full_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, fill_value, dtype, ...` | `Any` |  |
| `torch._numpy.gcd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.geomspace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `start, stop, num, ...` | `Any` |  |
| `torch._numpy.gradient` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f, varargs, axis, ...` | `Any` |  |
| `torch._numpy.greater` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.greater_equal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.hamming` | ❓ | ❓ | ❓ | ❓ | 🔴 | `M` | `Any` |  |
| `torch._numpy.hanning` | ❓ | ❓ | ❓ | ❓ | 🔴 | `M` | `Any` |  |
| `torch._numpy.heaviside` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.histogram` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, bins, range, ...` | `Any` |  |
| `torch._numpy.histogram2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, y, bins, ...` | `Any` |  |
| `torch._numpy.histogramdd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `sample, bins, range, ...` | `Any` |  |
| `torch._numpy.hsplit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ary, indices_or_sections` | `Any` |  |
| `torch._numpy.hstack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tup, dtype, casting` | `Any` |  |
| `torch._numpy.hypot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.i0` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `Any` |  |
| `torch._numpy.identity` | ❓ | ❓ | ❓ | ❓ | 🔴 | `n, dtype, like` | `Any` |  |
| `torch._numpy.iinfo` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtyp` | `Any` |  |
| `torch._numpy.imag` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch._numpy.indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dimensions, dtype, sparse` | `Any` |  |
| `torch._numpy.inner` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `Any` |  |
| `torch._numpy.invert` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.isclose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, rtol, ...` | `Any` |  |
| `torch._numpy.iscomplex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `Any` |  |
| `torch._numpy.iscomplexobj` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `Any` |  |
| `torch._numpy.isfinite` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.isinf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.isnan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.isneginf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out` | `Any` |  |
| `torch._numpy.isposinf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out` | `Any` |  |
| `torch._numpy.isreal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `Any` |  |
| `torch._numpy.isrealobj` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `Any` |  |
| `torch._numpy.isscalar` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch._numpy.issubdtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arg1, arg2` | `Any` |  |
| `torch._numpy.kaiser` | ❓ | ❓ | ❓ | ❓ | 🔴 | `M, beta` | `Any` |  |
| `torch._numpy.kron` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `Any` |  |
| `torch._numpy.lcm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.ldexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.left_shift` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.less` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.less_equal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.linspace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `start, stop, num, ...` | `Any` |  |
| `torch._numpy.log` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.log10` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.log1p` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.log2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.logaddexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.logaddexp2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.logical_and` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.logical_not` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.logical_or` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.logical_xor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.logspace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `start, stop, num, ...` | `Any` |  |
| `torch._numpy.matmul` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.max` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, out, ...` | `Any` |  |
| `torch._numpy.maximum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.mean` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, dtype, ...` | `Any` |  |
| `torch._numpy.median` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, out, ...` | `Any` |  |
| `torch._numpy.meshgrid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `xi, copy, sparse, ...` | `Any` |  |
| `torch._numpy.min` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, out, ...` | `Any` |  |
| `torch._numpy.min_scalar_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch._numpy.minimum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.mod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.modf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, args, kwds` | `Any` |  |
| `torch._numpy.moveaxis` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, source, destination` | `Any` |  |
| `torch._numpy.multiply` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.nan_to_num` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, copy, nan, ...` | `Any` |  |
| `torch._numpy.ndim` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch._numpy.negative` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.nextafter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.nonzero` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch._numpy.not_equal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.ones` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, dtype, order, ...` | `Any` |  |
| `torch._numpy.ones_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dtype, order, ...` | `Any` |  |
| `torch._numpy.outer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `Any` |  |
| `torch._numpy.pad` | ❓ | ❓ | ❓ | ❓ | 🔴 | `array, pad_width, mode, ...` | `Any` |  |
| `torch._numpy.percentile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, q, axis, ...` | `Any` |  |
| `torch._numpy.positive` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.power` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.prod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, dtype, ...` | `Any` |  |
| `torch._numpy.product` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, dtype, ...` | `Any` |  |
| `torch._numpy.ptp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, out, ...` | `Any` |  |
| `torch._numpy.put` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, indices, values, ...` | `Any` |  |
| `torch._numpy.put_along_axis` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arr, indices, values, ...` | `Any` |  |
| `torch._numpy.quantile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, q, axis, ...` | `Any` |  |
| `torch._numpy.rad2deg` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.radians` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.ravel` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, order` | `Any` |  |
| `torch._numpy.real` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch._numpy.real_if_close` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, tol` | `Any` |  |
| `torch._numpy.reciprocal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.remainder` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.repeat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, repeats, axis` | `Any` |  |
| `torch._numpy.reshape` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, newshape, order` | `Any` |  |
| `torch._numpy.resize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, new_shape` | `Any` |  |
| `torch._numpy.result_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arrays_and_dtypes` | `Any` |  |
| `torch._numpy.right_shift` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.rint` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.roll` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, shift, axis` | `Any` |  |
| `torch._numpy.rollaxis` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, start` | `Any` |  |
| `torch._numpy.rot90` | ❓ | ❓ | ❓ | ❓ | 🔴 | `m, k, axes` | `Any` |  |
| `torch._numpy.round` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, decimals, out` | `Any` |  |
| `torch._numpy.round_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, decimals, out` | `Any` |  |
| `torch._numpy.row_stack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tup, dtype, casting` | `Any` |  |
| `torch._numpy.searchsorted` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, v, side, ...` | `Any` |  |
| `torch._numpy.set_default_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fp_dtype, int_dtype` | `Any` | Set the (global) defaults for fp, complex, and int dtypes. The complex dtype is inferred from the float (fp) dtype. It has a width at least twice the width of the float dtype, i.e., it's complex128... |
| `torch._numpy.shape` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch._numpy.sign` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.signbit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.sin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.sinc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `Any` |  |
| `torch._numpy.sinh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.size` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis` | `Any` |  |
| `torch._numpy.sometrue` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, out, ...` | `Any` |  |
| `torch._numpy.sort` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, kind, ...` | `Any` |  |
| `torch._numpy.split` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ary, indices_or_sections, axis` | `Any` |  |
| `torch._numpy.sqrt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.square` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.squeeze` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis` | `Any` |  |
| `torch._numpy.stack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arrays, axis, out, ...` | `Any` |  |
| `torch._numpy.std` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, dtype, ...` | `Any` |  |
| `torch._numpy.subtract` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.sum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, dtype, ...` | `Any` |  |
| `torch._numpy.swapaxes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis1, axis2` | `Any` |  |
| `torch._numpy.take` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, indices, axis, ...` | `Any` |  |
| `torch._numpy.take_along_axis` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arr, indices, axis` | `Any` |  |
| `torch._numpy.tan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.tanh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.tensordot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, axes` | `Any` |  |
| `torch._numpy.tile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `A, reps` | `Any` |  |
| `torch._numpy.trace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, offset, axis1, ...` | `Any` |  |
| `torch._numpy.transpose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axes` | `Any` |  |
| `torch._numpy.tri` | ❓ | ❓ | ❓ | ❓ | 🔴 | `N, M, k, ...` | `Any` |  |
| `torch._numpy.tril` | ❓ | ❓ | ❓ | ❓ | 🔴 | `m, k` | `Any` |  |
| `torch._numpy.tril_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `n, k, m` | `Any` |  |
| `torch._numpy.tril_indices_from` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arr, k` | `Any` |  |
| `torch._numpy.triu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `m, k` | `Any` |  |
| `torch._numpy.triu_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `n, k, m` | `Any` |  |
| `torch._numpy.triu_indices_from` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arr, k` | `Any` |  |
| `torch._numpy.true_divide` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.trunc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.unique` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ar, return_index, return_inverse, ...` | `Any` |  |
| `torch._numpy.vander` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, N, increasing` | `Any` |  |
| `torch._numpy.var` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, dtype, ...` | `Any` |  |
| `torch._numpy.vdot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `Any` |  |
| `torch._numpy.vsplit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ary, indices_or_sections` | `Any` |  |
| `torch._numpy.vstack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tup, dtype, casting` | `Any` |  |
| `torch._numpy.where` | ❓ | ❓ | ❓ | ❓ | 🔴 | `condition, x, y` | `Any` |  |
| `torch._numpy.zeros` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, dtype, order, ...` | `Any` |  |
| `torch._numpy.zeros_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dtype, order, ...` | `Any` |  |
| `torch._prims.TensorMeta` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensorlike, shape, strides, ...` | `Any` |  |
| `torch._prims.backwards_not_supported` | ❓ | ❓ | ❓ | ❓ | 🔴 | `prim` | `Any` |  |
| `torch._prims.expand_dims` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dimensions, ndim` | `<class 'torch.Tensor'>` | Creates a view of a with a.ndim + len(dimensions) dimensions, with new dimensions of length one at the dimensions specified by dimensions. |
| `torch._prims.handle_torch_function` | ❓ | ❓ | ❓ | ❓ | 🔴 | `public_api, relevant_args, args, ...` | `typing.Any` | Implement a function with checks for ``__torch_function__`` overrides. See torch::autograd::handle_torch_function for the equivalent of this function in the C++ implementation. Arguments --------- ... |
| `torch._prims.is_functional_schema` | ❓ | ❓ | ❓ | ❓ | 🔴 | `schema` | `<class 'bool'>` | Check if the schema is functional. An operator is functional if: - it does not mutate any of its inputs - it does not return a view on any of its inputs - it has at least one return |
| `torch._prims.new_token_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'torch.Tensor'>` |  |
| `torch._prims.register_debug_prims` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._prims.register_rng_prims` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._prims.shift_right_logical` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` |  |
| `torch._prims.sym_float` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` | SymInt-aware utility for float casting. Args: a (SymInt, SymFloat, or object): Object to cast |
| `torch._prims.torch_var` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, correction, ...` | `Any` |  |
| `torch._prims.tree_flatten` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tree, is_leaf` | `tuple[list[typing.Any], torch.utils._pytree.TreeSpec]` | Flattens a pytree into a list of values and a TreeSpec that can be used to reconstruct the pytree. |
| `torch._prims.tree_map` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, tree, rests, ...` | `typing.Any` | Map a multi-input function over pytree args to produce a new pytree. See also :func:`tree_map_`. >>> tree_map(lambda x: x + 1, {'x': 7, 'y': (42, 64)}) {'x': 8, 'y': (43, 65)} >>> tree_map(lambda x... |
| `torch._prims.tree_unflatten` | ❓ | ❓ | ❓ | ❓ | 🔴 | `leaves, treespec` | `typing.Any` | Given a list of values and a TreeSpec, builds a pytree. This is the inverse operation of `tree_flatten`. |
| `torch._prims.type_to_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typ` | `torch.dtype` | Computes the corresponding dtype for a Number type. |
| `torch._prims_common.NamedTuple` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typename, fields, kwargs` | `Any` | Typed version of namedtuple. Usage:: class Employee(NamedTuple): name: str id: int This is equivalent to:: Employee = collections.namedtuple('Employee', ['name', 'id']) The resulting class has an e... |
| `torch._prims_common.alert_not_deterministic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `caller` | `Any` |  |
| `torch._prims_common.apply_perm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `inp, perm` | `Any` |  |
| `torch._prims_common.are_strides_like_channels_last` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, strides` | `bool` |  |
| `torch._prims_common.can_safe_cast_to` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cast_to, cast_from` | `bool` |  |
| `torch._prims_common.canonicalize_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `torch.device` |  |
| `torch._prims_common.canonicalize_dim` | ❓ | ❓ | ❓ | ❓ | 🔴 | `rank, idx, wrap_scalar` | `int` |  |
| `torch._prims_common.canonicalize_dims` | ❓ | ❓ | ❓ | ❓ | 🔴 | `rank, indices, wrap_scalar` | `Any` |  |
| `torch._prims_common.cast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typ, val` | `Any` | Cast a value to a type. This returns the value unchanged. To the type checker this signals that the return value has the designated type, but at runtime we intentionally don't check anything (we wa... |
| `torch._prims_common.check` | ❓ | ❓ | ❓ | ❓ | 🔴 | `b, s, exc_type` | `None` | Helper function for raising an error_type (default: RuntimeError) if a boolean condition fails. Error message is a callable producing a string (to avoid wasting time string formatting in non-error ... |
| `torch._prims_common.check_all_strides` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, only_cuda` | `tuple[bool, Optional[int]]` |  |
| `torch._prims_common.check_fp_or_complex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype, fn_name, allow_low_precision_dtypes` | `Any` | Checks whether the input is floating point or complex. If allow_low_precision_dtypes is True, it allows having float16, bfloat16, and complex32 |
| `torch._prims_common.check_in_bounds_for_storage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, shape, strides, ...` | `Any` | Determines if the given shape, strides, and offset are valid for the given storage. |
| `torch._prims_common.check_is_matrix` | ❓ | ❓ | ❓ | ❓ | 🔴 | `A, f_name, arg_name` | `Any` |  |
| `torch._prims_common.check_layout` | ❓ | ❓ | ❓ | ❓ | 🔴 | `layout` | `Any` |  |
| `torch._prims_common.check_pin_memory` | ❓ | ❓ | ❓ | ❓ | 🔴 | `pin_memory` | `Any` |  |
| `torch._prims_common.check_same_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, allow_cpu_scalar_tensors` | `Any` | Checks that all Tensors in args have the same device. Raises a RuntimeError when: - args contains an object whose type is not Tensor or Number - two Tensor objects in args have different devices, u... |
| `torch._prims_common.check_same_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args` | `Any` | Checks that all Tensors in args have the same device and that all Numbers have the same corresponding Python type. Raises a RuntimeError when: - args contains an object whose type is not Tensor or ... |
| `torch._prims_common.check_same_shape` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, allow_cpu_scalar_tensors` | `Any` | Checks that all Tensors in args have the same shape. Raises a RuntimeError when: - args contains an object whose type is not Tensor or Number - two Tensor objects in args have different devices |
| `torch._prims_common.check_significant_strides` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, only_cuda, ...` | `tuple[bool, Optional[int]]` |  |
| `torch._prims_common.clone_preserve_strides` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `Any` |  |
| `torch._prims_common.compare_tensor_meta` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, check_sizes, ...` | `Any` | Checks that two tensor likes have the same shape, dtype and device. In the future this will validate additional metadata, like strides. |
| `torch._prims_common.compute_elementwise_output_logical_to_physical_perm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, _skip_checks` | `list[int]` |  |
| `torch._prims_common.compute_elementwise_output_strides` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `tuple[int, ...]` | Computes the output strides for elementwise operations. |
| `torch._prims_common.compute_reduction_output_shape` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, dimensions` | `tuple[int, ...]` |  |
| `torch._prims_common.compute_required_storage_length` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, strides, storage_offset` | `int` | Computes the minimum storage size to hold the given tensor geometry. Example ======= This is the size of a newly allocated tensor's storage, in units of elements >>> t = torch.empty((10, 20)) >>> c... |
| `torch._prims_common.corresponding_complex_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `torch.dtype` |  |
| `torch._prims_common.corresponding_real_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `torch.dtype` |  |
| `torch._prims_common.device_or_default` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `DeviceLikeType` |  |
| `torch._prims_common.dtype_or_default` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `torch.dtype` |  |
| `torch._prims_common.dtype_to_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `type` | Computes the corresponding Python type (AKA "type kind") for the given dtype. |
| `torch._prims_common.dtype_to_type_ctor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `Callable[[NumberType], NumberType]` | Computes the corresponding Python type constructor for the given dtype. |
| `torch._prims_common.elementwise_dtypes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `_args, type_promotion_kind` | `tuple[torch.dtype, torch.dtype]` | Computes the computation and result dtypes for elementwise type promotion on the given arguments and with the given elementwise type promotion kind. Note that not all inputs to an elementwise opera... |
| `torch._prims_common.expr_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `type` |  |
| `torch._prims_common.extract_dims_from_varargs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dims` | `DimsSequenceType` |  |
| `torch._prims_common.extract_shape` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, allow_cpu_scalar_tensors` | `Optional[ShapeType]` |  |
| `torch._prims_common.extract_shape_from_varargs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, validate` | `tuple[int, ...]` | Returns a shape from varargs. In PyTorch, operations that accept shapes often accept them as varargs, like foo(*shape). However a user can pass the shape as a sequence of integers, like this: foo(1... |
| `torch._prims_common.get_acc_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype, device` | `torch.dtype` |  |
| `torch._prims_common.get_aten_op` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn, name` | `Any` | Given the __module__ of reference and its name, it returns (our best guess of) the ATen name of the associated operation Note: In ATen, the __name__ of a function within a module often starts by th... |
| `torch._prims_common.get_computation_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `torch.dtype` |  |
| `torch._prims_common.get_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `Any` |  |
| `torch._prims_common.get_higher_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `Optional[torch.dtype]` | Computes the "lowest" datatype that is weakly "higher" than both a and b. |
| `torch._prims_common.get_higher_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `type` | Returns the higher of the two given Number types. The types are ordered bool -> int -> float -> complex. |
| `torch._prims_common.infer_size` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, numel` | `tuple[int, ...]` | Infers the size of a dim with size -1, if it exists. Also checks that new shape is compatible with the number of elements. |
| `torch._prims_common.infer_size_shapes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `tuple[int, ...]` |  |
| `torch._prims_common.invert_perm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `perm` | `Any` |  |
| `torch._prims_common.is_boolean_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `bool` |  |
| `torch._prims_common.is_channels_last_contiguous` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `bool` | True when a tensor is channels-last contiguous. This requires that: - the tensor is conceptually either 4 (NHWC) or 5 (NDHWC) dimensions - if we name the tensor's dimensions NCHW or NCDHW, then the... |
| `torch._prims_common.is_channels_last_contiguous_2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `bool` |  |
| `torch._prims_common.is_channels_last_contiguous_3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `bool` |  |
| `torch._prims_common.is_complex_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `bool` |  |
| `torch._prims_common.is_contiguous` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `bool` | Tests whether a tensor is contiguous or not. Tensors are contiguous when they have no elements, one element, or when they have "nested" strides. |
| `torch._prims_common.is_contiguous_for_memory_format` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, memory_format` | `bool` |  |
| `torch._prims_common.is_cpu_scalar_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `bool` |  |
| `torch._prims_common.is_expandable_to` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, desired` | `bool` | Checks if a shape can be expanded to another shape. This is equivalent to checking if the two shapes are broadcastable. |
| `torch._prims_common.is_float_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `bool` |  |
| `torch._prims_common.is_grad_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `bool` | Checks if the dtype can require a gradient. |
| `torch._prims_common.is_integer_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `bool` |  |
| `torch._prims_common.is_low_precision_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `bool` |  |
| `torch._prims_common.is_non_overlapping_and_dense` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `bool` | True when a tensor is non-overlapping and dense. A tensor is non-overlapping and dense when there exists a permutation of its dimensions that is contiguous. |
| `torch._prims_common.is_same_shape` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `bool` | Compares two shapes a and b, returning True if they are the same (their ranks and corresponding lengths match) and False otherwise. |
| `torch._prims_common.is_valid_permutation` | ❓ | ❓ | ❓ | ❓ | 🔴 | `rank, perm` | `bool` | Validates that perm is a permutation of length rank. |
| `torch._prims_common.is_weakly_lesser_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `bool` | Compares two types, a and b, returning True if a is weakly "less" than b. The comparison is determined by the following type ordering: bool, int, float, complex. |
| `torch._prims_common.layout_or_default` | ❓ | ❓ | ❓ | ❓ | 🔴 | `layout` | `torch.layout` |  |
| `torch._prims_common.make_channels_last_1d_strides_for` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape` | `tuple[Union[_IntLikeT, int], ...]` |  |
| `torch._prims_common.make_channels_last_2d_strides_for` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape` | `tuple[Union[_IntLikeT, int], ...]` |  |
| `torch._prims_common.make_channels_last_3d_strides_for` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape` | `tuple[Union[_IntLikeT, int], ...]` |  |
| `torch._prims_common.make_channels_last_strides_for` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape` | `tuple[Union[_IntLikeT, int], ...]` |  |
| `torch._prims_common.make_contiguous_strides_for` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, row_major` | `tuple[Union[_IntLikeT, int], ...]` | Returns the strides of a contiguous tensor if row_major If row_major=True, it returns the strides of a contiguous batch of Fortran-contiguous matrices This is often used when calling external libra... |
| `torch._prims_common.mask_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mask, t` | `Any` | Similar to torch.where(mask, t, 0) but if t is boolean, result is also boolean and not promoted to int. |
| `torch._prims_common.number_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `type` |  |
| `torch._prims_common.overload` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` | Decorator for overloaded functions/methods. In a stub file, place two or more stub definitions for the same function in a row, each decorated with @overload. For example:: @overload def utf8(value:... |
| `torch._prims_common.prod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `xs` | `NumberType` | Product of elements in input sequence. Returns 1 for empty sequence |
| `torch._prims_common.reduction_dims` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, dims` | `tuple[int, ...]` |  |
| `torch._prims_common.reduction_dtypes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arg, output_dtype_kind, dtype` | `tuple[torch.dtype, Optional[torch.dtype]]` |  |
| `torch._prims_common.same_shape` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, allow_rhs_unbacked` | `bool` |  |
| `torch._prims_common.set_correction` | ❓ | ❓ | ❓ | ❓ | 🔴 | `unbiased, correction` | `float` |  |
| `torch._prims_common.suggest_memory_format` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `torch.memory_format` |  |
| `torch._prims_common.sym_float` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` | SymInt-aware utility for float casting. Args: a (SymInt, SymFloat, or object): Object to cast |
| `torch._prims_common.sym_int` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` | SymInt-aware utility for int casting. Args: a (SymInt, SymFloat, or object): Object to cast |
| `torch._prims_common.sym_max` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `Any` | SymInt-aware utility for max which avoids branching on a < b. Unlike builtins.max(), this only works for int/float, and it always promotes to float if any argument is float (unlike builtins.max, wh... |
| `torch._prims_common.type_to_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typ` | `torch.dtype` | Computes the corresponding dtype for a Number type. |
| `torch._prims_common.validate_dim_length` | ❓ | ❓ | ❓ | ❓ | 🔴 | `length` | `Any` | Validates that an object represents a valid dimension length. |
| `torch._prims_common.validate_dimension_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `rank, indices` | `Any` |  |
| `torch._prims_common.validate_exclusive_idx` | ❓ | ❓ | ❓ | ❓ | 🔴 | `rank, ex_idx` | `Any` | Validates that ex_idx is a valid exclusive index for the given shape. |
| `torch._prims_common.validate_idx` | ❓ | ❓ | ❓ | ❓ | 🔴 | `rank, idx` | `Any` | Validates that idx is a valid index for the given shape. Assumes the index is already canonicalized. |
| `torch._prims_common.validate_memory_format` | ❓ | ❓ | ❓ | ❓ | 🔴 | `memory_format` | `Any` |  |
| `torch._prims_common.validate_no_repeating_dims` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dims` | `Any` |  |
| `torch._prims_common.validate_shape` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape` | `Any` | Validates that a sequence represents a valid shape. |
| `torch._prims_common.validate_strides` | ❓ | ❓ | ❓ | ❓ | 🔴 | `strides` | `Any` | Verifies the object specifies valid strides. |
| `torch._refs.T` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `<class 'torch.Tensor'>` |  |
| `torch._refs.abs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.abs_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.acos` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.acos_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.acosh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.acosh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.add` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, alpha, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.add |
| `torch._refs.add_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, alpha, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.add |
| `torch._refs.addcdiv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, tensor1, tensor2, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.addcdiv |
| `torch._refs.addcdiv_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, tensor1, tensor2, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.addcdiv |
| `torch._refs.addcmul` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, tensor1, tensor2, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.addcmul |
| `torch._refs.addcmul_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, tensor1, tensor2, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.addcmul |
| `torch._refs.addr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, vec1, vec2, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.alias` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `<class 'torch.Tensor'>` |  |
| `torch._refs.alias_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, out, kwargs` | `<class 'torch.Tensor'>` |  |
| `torch._refs.all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, keepdim, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.allclose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, rtol, ...` | `<class 'bool'>` | Reference implementation of torch.allclose |
| `torch._refs.amax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, keepdim, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.amin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, keepdim, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, keepdim, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.arange` | ❓ | ❓ | ❓ | ❓ | 🔴 | `start, end, step, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.as_strided` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, size, stride, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.as_strided_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, out, kwargs` | `<class 'torch.Tensor'>` |  |
| `torch._refs.as_strided_scatter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, src, size, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.asin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.asin_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.asinh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.asinh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.atan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.atan2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.atan2_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.atan_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.atanh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.atanh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.atleast_1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arg, args` | `typing.Union[torch.Tensor, tuple[torch.Tensor, ...]]` | Reference implementation of :func:`torch.atleast_1d`. |
| `torch._refs.atleast_2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arg, args` | `typing.Union[torch.Tensor, tuple[torch.Tensor, ...]]` | Reference implementation of :func:`torch.atleast_2d`. |
| `torch._refs.atleast_3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arg, args` | `typing.Union[torch.Tensor, tuple[torch.Tensor, ...]]` | Reference implementation of :func:`torch.atleast_3d`. |
| `torch._refs.bitwise_and` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.bitwise_and_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.bitwise_left_shift` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.bitwise_left_shift_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.bitwise_not` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.bitwise_not_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.bitwise_or` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.bitwise_or_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.bitwise_right_shift` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.bitwise_right_shift_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.bitwise_xor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.bitwise_xor_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.block_diag` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `<class 'torch.Tensor'>` | This is used as an input to PythonRefInfo. `torch.block_diag` expects arguments splatted, but `aten.block_diag` expects only one argument that is a list of Tensors. |
| `torch._refs.broadcast_shapes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shapes` | `typing.Union[torch.Size, list[int], tuple[int, ...]]` |  |
| `torch._refs.broadcast_tensors` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `list[torch.Tensor]` |  |
| `torch._refs.broadcast_to` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, size` | `<class 'torch.Tensor'>` |  |
| `torch._refs.bucketize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, boundaries, out_int32, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.cast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typ, val` | `Any` | Cast a value to a type. This returns the value unchanged. To the type checker this signals that the return value has the designated type, but at runtime we intentionally don't check anything (we wa... |
| `torch._refs.cat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, dim, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.cauchy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, median, sigma, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.cauchy_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, median, sigma, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.ceil` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.ceil_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.chunk` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, chunks, dim` | `tuple[torch.Tensor, ...]` |  |
| `torch._refs.clamp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, min, max, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.clamp_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, min, max, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.clamp_max` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, max, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.clamp_max_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, max, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.clamp_min` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, min, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.clamp_min_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, min, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.clone` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, memory_format, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.column_stack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.conj` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input` | `<class 'torch.Tensor'>` |  |
| `torch._refs.conj_physical` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.conj_physical_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.constant_pad_nd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, pad, value, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.contiguous` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, memory_format` | `<class 'torch.Tensor'>` |  |
| `torch._refs.copy_to` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, allow_cross_device` | `Any` |  |
| `torch._refs.copysign` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.copysign_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.cos` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.cos_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.cosh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.cosh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.count_nonzero` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, dim, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.cumprod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, dtype, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.cumprod_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, dtype, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.cumsum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, dtype, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.cumsum_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, dtype, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.deg2rad` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.deg2rad_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.diag` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, offset, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.diag_embed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `t, offset, dim1, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.diag_embed |
| `torch._refs.diagonal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, offset, dim1, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.diagonal |
| `torch._refs.diagonal_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, out, kwargs` | `<class 'torch.Tensor'>` |  |
| `torch._refs.diagonal_scatter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, src, offset, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.digamma` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.digamma_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.div` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, rounding_mode, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.div |
| `torch._refs.div_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, rounding_mode, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.div |
| `torch._refs.dot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, other, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.dsplit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, sections` | `typing.Union[list[torch.Tensor], tuple[torch.Tensor, ...]]` |  |
| `torch._refs.dstack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.dtype_to_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `type` | Computes the corresponding Python type (AKA "type kind") for the given dtype. |
| `torch._refs.elementwise_unary_scalar_wrapper` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `typing.Callable[~_P, typing.Union[~_T, bool, int, float, complex]]` | Allows unary operators that accept tensors to work with Python numbers. |
| `torch._refs.empty` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, dtype, layout, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.empty_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dtype, device, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.empty_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size, out, memory_format` | `<class 'torch.Tensor'>` |  |
| `torch._refs.empty_permuted` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, physical_layout, dtype, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.empty_strided` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, strides, dtype, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.eq` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.eq_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.equal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `<class 'bool'>` |  |
| `torch._refs.erf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.erf_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.erfc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.erfc_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.erfinv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.erfinv_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.exp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.exp2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.exp2_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.exp_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.expand` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, shape` | `<class 'torch.Tensor'>` |  |
| `torch._refs.expand_as` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `<class 'torch.Tensor'>` |  |
| `torch._refs.expand_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, out, kwargs` | `<class 'torch.Tensor'>` |  |
| `torch._refs.expm1` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.expm1_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.exponential` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, rate, generator, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.exponential_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, rate, generator, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.eye` | ❓ | ❓ | ❓ | ❓ | 🔴 | `n, m, dtype, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.eye |
| `torch._refs.fill` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, value, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.fill_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, value` | `<class 'torch.Tensor'>` |  |
| `torch._refs.flatten` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, start_dim, end_dim` | `<class 'torch.Tensor'>` |  |
| `torch._refs.flip` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dims, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.fliplr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `<class 'torch.Tensor'>` |  |
| `torch._refs.flipud` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `<class 'torch.Tensor'>` |  |
| `torch._refs.float_power` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.float_power_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.floor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.floor_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.floor_divide` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.floor_divide_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.fmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.fmin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.fmod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.fmod_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.frac` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.frac_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.frexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, out` | `<class 'torch._prims_common.wrappers.return_types_frexp'>` |  |
| `torch._refs.full` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, fill_value, dtype, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.full_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, fill_value, dtype, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.gcd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.gcd_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.ge` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.ge_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.geometric` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, p, generator, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.geometric_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, p, generator, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.gt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.gt_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.handle_noncontiguous_outputs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input_tlist, output` | `Any` |  |
| `torch._refs.heaviside` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.heaviside_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.hsplit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, indices_or_sections` | `tuple[torch.Tensor, ...]` |  |
| `torch._refs.hstack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.hypot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.hypot_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.i0` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.i0_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.igamma` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.igamma_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.igammac` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.igammac_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.imag` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `<class 'torch.Tensor'>` |  |
| `torch._refs.index_add` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, dim, index, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.index_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, dim, index, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.index_copy_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, dim, index, ...` | `Any` |  |
| `torch._refs.index_fill` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, dim, index, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.index_fill_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, dim, index, ...` | `Any` |  |
| `torch._refs.index_select` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, dim, index, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.is_complex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input` | `Any` |  |
| `torch._refs.is_noncontiguous_supported` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `Any` |  |
| `torch._refs.is_weakly_lesser_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `bool` | Compares two types, a and b, returning True if a is weakly "less" than b. The comparison is determined by the following type ordering: bool, int, float, complex. |
| `torch._refs.isclose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, rtol, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.isfinite` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.isinf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.isnan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.isneginf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.isposinf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.isreal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.istft` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, n_fft, hop_length, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.item` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `typing.Union[bool, int, float, complex]` |  |
| `torch._refs.lcm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.lcm_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.le` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.le_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.lerp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `start, end, weight, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.lerp_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `start, end, weight, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.lgamma` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.lgamma_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.linspace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `start, end, steps, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.log` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.log10` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.log10_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.log1p` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.log1p_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.log2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.log2_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.log_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.log_normal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, mean, std, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.log_normal_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, mean, std, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.log_softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, dtype, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.logaddexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.logaddexp2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.logical_and` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.logical_and_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.logical_not` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.logical_not_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.logical_or` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.logical_or_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.logical_xor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.logical_xor_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.logspace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `start, end, steps, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.logsumexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, dim, keepdim, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.lt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.lt_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.masked_fill` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, mask, value, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.masked_fill_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, mask, value` | `<class 'torch.Tensor'>` |  |
| `torch._refs.maximum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.mean` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, keepdim, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.meshgrid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, indexing` | `list[torch.Tensor]` |  |
| `torch._refs.minimum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.movedim` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, source, destination` | `<class 'torch.Tensor'>` | Reference implementation of torch.movedim |
| `torch._refs.mul` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.mul_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.mvlgamma` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` |  |
| `torch._refs.mvlgamma_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` |  |
| `torch._refs.nan_to_num` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, nan, posinf, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.nan_to_num_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, nan, posinf, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.narrow` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, start, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.narrow_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, out, kwargs` | `<class 'torch.Tensor'>` |  |
| `torch._refs.native_group_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, weight, bias, ...` | `tuple[torch.Tensor, torch.Tensor, torch.Tensor]` |  |
| `torch._refs.native_layer_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, normalized_shape, weight, ...` | `<class 'torch._prims_common.wrappers.return_types_native_layer_norm'>` |  |
| `torch._refs.native_layer_norm_fake` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fake_mode, func, args, ...` | `Any` |  |
| `torch._refs.ne` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.ne_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.neg` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.neg_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.new_empty` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, size, dtype, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.new_empty_strided` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, size, stride, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.Tensor.new_empty_strided |
| `torch._refs.new_full` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, size, fill_value, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.new_ones` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, size, dtype, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.new_zeros` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, size, dtype, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.nextafter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.nextafter_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, p, dim, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.normal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mean, std, size, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.normal_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, mean, std, ...` | `Any` |  |
| `torch._refs.ones` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size, dtype, layout, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.ones_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dtype, layout, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.out_wrapper` | ❓ | ❓ | ❓ | ❓ | 🔴 | `out_names, exact_dtype, pass_is_out, ...` | `typing.Callable[[typing.Callable[~_P, ~_T]], typing.Callable[~_P, ~_T]]` |  |
| `torch._refs.overload` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` | Decorator for overloaded functions/methods. In a stub file, place two or more stub definitions for the same function in a row, each decorated with @overload. For example:: @overload def utf8(value:... |
| `torch._refs.permute` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dims` | `<class 'torch.Tensor'>` |  |
| `torch._refs.permute_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, out, kwargs` | `<class 'torch.Tensor'>` |  |
| `torch._refs.positive` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `<class 'torch.Tensor'>` |  |
| `torch._refs.pow` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.pow_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.prod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, keepdim, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.rad2deg` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.rad2deg_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.randn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, dtype, device, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.ravel` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `<class 'torch.Tensor'>` |  |
| `torch._refs.real` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `<class 'torch.Tensor'>` |  |
| `torch._refs.reciprocal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.reciprocal_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.register_decomposition` | ❓ | ❓ | ❓ | ❓ | 🔴 | `aten_op, registry, type, ...` | `typing.Callable[[typing.Callable[~_P, ~_T]], typing.Callable[~_P, ~_T]]` | A decorator to register a function as a decomposition to the Python decomposition table. Use it like this:: @register_decomposition(torch.ops.aten.clamp_min) def clamp_min(x): return torch.clamp(se... |
| `torch._refs.remainder` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.remainder_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.renorm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, p, dim, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.repeat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, repeat_shape, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.reshape` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, shape` | `<class 'torch.Tensor'>` |  |
| `torch._refs.reshape_as` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, other` | `<class 'torch.Tensor'>` |  |
| `torch._refs.rfloordiv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `<class 'torch.Tensor'>` |  |
| `torch._refs.roll` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, shifts, dims, ...` | `<class 'torch.Tensor'>` | Reference implementation of :func:`torch.roll`. |
| `torch._refs.rot90` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, k, dims, ...` | `<class 'torch.Tensor'>` | Reference implementation of :func:`torch.rot90`. |
| `torch._refs.round` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, decimals, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.rpow` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `<class 'torch.Tensor'>` |  |
| `torch._refs.rsqrt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.rsqrt_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.rsub` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, alpha, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.rtruediv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `<class 'torch.Tensor'>` |  |
| `torch._refs.scalar_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dtype, layout, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.select_scatter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, src, dim, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sgn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sgn_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sigmoid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sigmoid_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sign` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sign_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.signbit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sin_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sinc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sinc_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.singledispatch` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` | Single-dispatch generic function decorator. Transforms a function into a generic function, which can have different behaviours depending upon the type of its first argument. The decorated function ... |
| `torch._refs.sinh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sinh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, dtype, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.split_with_sizes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, split_sizes, dim` | `list[torch.Tensor]` |  |
| `torch._refs.sqrt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sqrt_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.square` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.square_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.squeeze` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim` | `<class 'torch.Tensor'>` |  |
| `torch._refs.squeeze_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, out, kwargs` | `<class 'torch.Tensor'>` |  |
| `torch._refs.stack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, dim, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.std` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, unbiased, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.std_mean` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, unbiased, ...` | `<class 'torch._prims_common.wrappers.return_types_std_mean'>` |  |
| `torch._refs.stft` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, n_fft, hop_length, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sub` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, alpha, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.sub |
| `torch._refs.sub_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, alpha, ...` | `<class 'torch.Tensor'>` | Reference implementation of torch.sub |
| `torch._refs.sum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, keepdim, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sum_to_size` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, shape` | `<class 'torch.Tensor'>` |  |
| `torch._refs.swap_axes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim0, dim1` | `<class 'torch.Tensor'>` |  |
| `torch._refs.sym_float` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` | SymInt-aware utility for float casting. Args: a (SymInt, SymFloat, or object): Object to cast |
| `torch._refs.sym_int` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` | SymInt-aware utility for int casting. Args: a (SymInt, SymFloat, or object): Object to cast |
| `torch._refs.t` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch._refs.t_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, out, kwargs` | `<class 'torch.Tensor'>` |  |
| `torch._refs.take_along_dim` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, indices, dim, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.tan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.tan_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.tanh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.tanh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `data, dtype, device, ...` | `Any` |  |
| `torch._refs.tensor_split` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, indices_or_sections, dim` | `tuple[torch.Tensor, ...]` |  |
| `torch._refs.to` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, args, kwargs` | `<class 'torch.Tensor'>` |  |
| `torch._refs.trace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.transpose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim0, dim1` | `<class 'torch.Tensor'>` |  |
| `torch._refs.transpose_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, out, kwargs` | `<class 'torch.Tensor'>` |  |
| `torch._refs.tril` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, diagonal, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.tril_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, diagonal, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.tril_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `row, col, offset, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.triu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, diagonal, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.triu_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, diagonal, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.triu_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `row, col, offset, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.true_divide` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.true_divide_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.trunc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.trunc_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.trunc_divide` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.unbind` | ❓ | ❓ | ❓ | ❓ | 🔴 | `t, dim` | `typing.Union[list[torch.Tensor], tuple[torch.Tensor, ...]]` |  |
| `torch._refs.unbind_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, out, kwargs` | `<class 'torch.Tensor'>` |  |
| `torch._refs.unflatten` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, sizes` | `<class 'torch.Tensor'>` |  |
| `torch._refs.unfold` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, dimension, size, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.unfold_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, dimension, size, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.unsqueeze` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim` | `<class 'torch.Tensor'>` |  |
| `torch._refs.unsqueeze_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, out, kwargs` | `<class 'torch.Tensor'>` |  |
| `torch._refs.var` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, unbiased, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.var_mean` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dim, unbiased, ...` | `<class 'torch._prims_common.wrappers.return_types_var_mean'>` |  |
| `torch._refs.vdot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, other, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.view` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, shape` | `<class 'torch.Tensor'>` |  |
| `torch._refs.view_as` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, other` | `<class 'torch.Tensor'>` |  |
| `torch._refs.view_as_complex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self` | `<class 'torch.Tensor'>` |  |
| `torch._refs.view_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, out, kwargs` | `<class 'torch.Tensor'>` |  |
| `torch._refs.vsplit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, indices_or_sections` | `tuple[torch.Tensor, ...]` |  |
| `torch._refs.vstack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.where` | ❓ | ❓ | ❓ | ❓ | 🔴 | `pred, a, b, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.wraps` | ❓ | ❓ | ❓ | ❓ | 🔴 | `wrapped, assigned, updated` | `Any` | Decorator factory to apply update_wrapper() to a wrapper function Returns a decorator that invokes update_wrapper() with the decorated function as the wrapper argument and the arguments to wraps() ... |
| `torch._refs.xlogy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.xlogy_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.zero` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.zero_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, out` | `<class 'torch.Tensor'>` |  |
| `torch._refs.zeros` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size, dtype, layout, ...` | `<class 'torch.Tensor'>` |  |
| `torch._refs.zeros_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dtype, layout, ...` | `<class 'torch.Tensor'>` |  |
| | | | | | | | | |
| 🟦 ACCELERATOR_SUPPORT | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.accelerator.current_accelerator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `check_available` | `typing.Optional[torch.device]` | Return the device of the accelerator available at compilation time. If no accelerator were available at compilation time, returns None. See :ref:`accelerator<accelerators>` for details. Args: check... |
| `torch.accelerator.current_device_idx` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the index of a currently selected device for the current :ref:`accelerator<accelerators>`. Returns: int: the index of a currently selected device. |
| `torch.accelerator.current_device_index` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the index of a currently selected device for the current :ref:`accelerator<accelerators>`. Returns: int: the index of a currently selected device. |
| `torch.accelerator.current_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.Stream'>` | Return the currently selected stream for a given device. Args: device (:class:`torch.device`, str, int, optional): a given device that must match the current :ref:`accelerator<accelerators>` device... |
| `torch.accelerator.device_count` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the number of current :ref:`accelerator<accelerators>` available. Returns: int: the number of the current :ref:`accelerator<accelerators>` available. If there is no available accelerators, r... |
| `torch.accelerator.is_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Check if the current accelerator is available at runtime: it was build, all the required drivers are available and at least one device is visible. See :ref:`accelerator<accelerators>` for details. ... |
| `torch.accelerator.set_device_idx` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Set the current device index to a given device. Args: device (:class:`torch.device`, str, int): a given device that must match the current :ref:`accelerator<accelerators>` device type. .. note:: Th... |
| `torch.accelerator.set_device_index` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Set the current device index to a given device. Args: device (:class:`torch.device`, str, int): a given device that must match the current :ref:`accelerator<accelerators>` device type. .. note:: Th... |
| `torch.accelerator.set_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `None` | Set the current stream to a given stream. Args: stream (torch.Stream): a given stream that must match the current :ref:`accelerator<accelerators>` device type. .. note:: This function will set the ... |
| `torch.accelerator.synchronize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Wait for all kernels in all streams on the given device to complete. Args: device (:class:`torch.device`, str, int, optional): device for which to synchronize. It must match the current :ref:`accel... |
| | | | | | | | | |
| 🟦 AUTOMATIC_MIXED_PRECISION | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.amp.custom_bwd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `bwd, device_type` | `Any` | Create a helper decorator for backward methods of custom autograd functions. Autograd functions are subclasses of :class:`torch.autograd.Function`. Ensures that ``backward`` executes with the same ... |
| `torch.amp.custom_fwd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fwd, device_type, cast_inputs` | `Any` | Create a helper decorator for ``forward`` methods of custom autograd functions. Autograd functions are subclasses of :class:`torch.autograd.Function`. See the :ref:`example page<amp-custom-examples... |
| `torch.amp.is_autocast_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device_type` | `<class 'bool'>` | Return a bool indicating if autocast is available on :attr:`device_type`. Args: device_type(str): Device type to use. Possible values are: 'cuda', 'cpu', 'mtia', 'maia', 'xpu', and so on. The type ... |
| | | | | | | | | |
| 🟦 AUTOGRAD | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.autograd.backward` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, grad_tensors, retain_graph, ...` | `None` | Compute the sum of gradients of given tensors with respect to graph leaves. The graph is differentiated using the chain rule. If any of ``tensors`` are non-scalar (i.e. their data has more than one... |
| `torch.autograd.cast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typ, val` | `Any` | Cast a value to a type. This returns the value unchanged. To the type checker this signals that the return value has the designated type, but at runtime we intentionally don't check anything (we wa... |
| `torch.autograd.grad` | ❓ | ❓ | ❓ | ❓ | 🔴 | `outputs, inputs, grad_outputs, ...` | `tuple[torch.Tensor, ...]` | Compute and return the sum of gradients of outputs with respect to the inputs. ``grad_outputs`` should be a sequence of length matching ``output`` containing the "vector" in vector-Jacobian product... |
| `torch.autograd.gradcheck` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, inputs, eps, ...` | `<class 'bool'>` | Check gradients computed via small finite differences against analytical gradients wrt tensors in :attr:`inputs` that are of floating point or complex type and with ``requires_grad=True``. The chec... |
| `torch.autograd.gradgradcheck` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, inputs, grad_outputs, ...` | `<class 'bool'>` | Check gradients of gradients computed via small finite differences against analytical gradients wrt tensors in :attr:`inputs` and :attr:`grad_outputs` that are of floating point or complex type and... |
| `torch.autograd.handle_torch_function` | ❓ | ❓ | ❓ | ❓ | 🔴 | `public_api, relevant_args, args, ...` | `typing.Any` | Implement a function with checks for ``__torch_function__`` overrides. See torch::autograd::handle_torch_function for the equivalent of this function in the C++ implementation. Arguments --------- ... |
| `torch.autograd.is_tensor_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `inp` | `Any` | Returns ``True`` if the passed-in input is a Tensor-like. Currently, this occurs whenever there's a ``__torch_function__`` attribute on the type of the input. Examples -------- A subclass of tensor... |
| `torch.autograd.variable` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` |  |
| | | | | | | | | |
| 🟦 BACKEND_MANAGEMENT | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.backends.contextmanager` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` | @contextmanager decorator. Typical usage: @contextmanager def some_generator(<arguments>): <setup> try: yield <value> finally: <cleanup> This makes this: with some_generator(<arguments>) as <variab... |
| `torch.backends.disable_global_flags` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.backends.flags_frozen` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| | | | | | | | | |
| 🟦 COMPILATION | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.compiler.allow_in_graph` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` | Tells the compiler frontend (Dynamo) to skip symbolic introspection of the function and instead directly write it to the graph when encountered. If you are using :func:`torch.compile` (with backend... |
| `torch.compiler.assume_constant_result` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` | This function is used to mark a function `fn` as having a constant result. This allows the compiler to optimize away your function. Returns The same function `fn` Args: fn: The function to be marke... |
| `torch.compiler.compile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | See :func:`torch.compile` for details on the arguments for this function. |
| `torch.compiler.cudagraph_mark_step_begin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Indicates that a new iteration of inference or training is about to begin. CUDA Graphs will free tensors of a prior iteration. A new iteration is started on each invocation of torch.compile, so lon... |
| `torch.compiler.disable` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn, recursive, reason` | `Any` | This function provides a decorator to disable compilation on a function. It also provides the option of recursively disabling called functions. Args: fn (optional): The function to disable recursiv... |
| `torch.compiler.is_compiling` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Indicates whether a graph is executed/traced as part of torch.compile() or torch.export(). Note that there are 2 other related flags that should deprecated eventually: * torch._dynamo.external_util... |
| `torch.compiler.is_dynamo_compiling` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Indicates whether a graph is traced via TorchDynamo. It's stricter than is_compiling() flag, as it would only be set to True when TorchDynamo is used. Example:: >>> def forward(self, x): >>> if not... |
| `torch.compiler.is_exporting` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Indicated whether we're under exporting. It's stricter than is_compiling() flag, as it would only be set to True when torch.export is used. Example:: >>> def forward(self, x): >>> if not torch.comp... |
| `torch.compiler.list_backends` | ❓ | ❓ | ❓ | ❓ | 🔴 | `exclude_tags` | `list[str]` | Return valid strings that can be passed to `torch.compile(..., backend="name")`. Args: exclude_tags(optional): A tuple of strings representing tags to exclude. |
| `torch.compiler.load_cache_artifacts` | ❓ | ❓ | ❓ | ❓ | 🔴 | `serialized_artifacts` | `typing.Optional[ForwardRef('CacheInfo')]` | Hot loads cache artifacts that were previously serialized via save_cache_artifacts Example: # From a previous invocation artifacts = torch.compiler.save_cache_artifacts() torch.compiler.load_cache_... |
| `torch.compiler.reset` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | This function clears all compilation caches and restores the system to its initial state. It is recommended to call this function, especially after using operations like `torch.compile(...)` to ens... |
| `torch.compiler.save_cache_artifacts` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `typing.Optional[tuple[bytes, 'CacheInfo']]` | Serializes all the cache artifacts that were created during the compilation Example: - Execute torch.compile - Call torch.compiler.save_cache_artifacts() |
| `torch.compiler.set_stance` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stance, skip_guard_eval_unsafe, force_backend` | `Any` | Set the current stance of the compiler. Can be used as a function, context manager, or decorator. Do not use this function inside a `torch.compile` region - an error will be raised otherwise. .. co... |
| `torch.compiler.substitute_in_graph` | ❓ | ❓ | ❓ | ❓ | 🔴 | `original_fn, can_constant_fold_through, skip_signature_check` | `typing.Callable[[typing.Callable[~_P, ~_R]], typing.Callable[~_P, ~_R]]` | Register a polyfill handler for a function, usually a C function from the C extension, to be used in place of the original function when inlining the original function in the graph. .. note:: The p... |
| `torch.compiler.wrap_numpy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` | Decorator that turns a function from ``np.ndarray``s to ``np.ndarray``s into a function from ``torch.Tensor``s to ``torch.Tensor``s. It is designed to be used with :func:`torch.compile` with ``full... |
| | | | | | | | | |
| 🟦 CPU_OPERATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.cpu.current_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'str'>` | Returns current device for cpu. Always 'cpu'. N.B. This function only exists to facilitate device-agnostic code |
| `torch.cpu.current_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.cpu.Stream'>` | Returns the currently selected :class:`Stream` for a given device. Args: device (torch.device or int, optional): Ignored. N.B. This function only exists to facilitate device-agnostic code |
| `torch.cpu.device_count` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Returns number of CPU devices (not cores). Always 1. N.B. This function only exists to facilitate device-agnostic code |
| `torch.cpu.is_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Returns a bool indicating if CPU is currently available. N.B. This function only exists to facilitate device-agnostic code |
| `torch.cpu.set_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Sets the current device, in CPU we do nothing. N.B. This function only exists to facilitate device-agnostic code |
| `torch.cpu.stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `<class 'contextlib.AbstractContextManager'>` | Wrapper around the Context-manager StreamContext that selects a given stream. N.B. This function only exists to facilitate device-agnostic code |
| `torch.cpu.synchronize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Waits for all kernels in all streams on the CPU device to complete. Args: device (torch.device or int, optional): ignored, there's only one CPU device. N.B. This function only exists to facilitate ... |

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
