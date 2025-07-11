# TorchDevice Comprehensive Migration Plan
*Complete Function Routing & Device Compatibility Matrix*

## Executive Summary

This document provides a complete roadmap for refactoring TorchDevice into a **function router system** that intelligently routes PyTorch operations across CPU, CUDA, MPS, and MLX devices based on compatibility and performance.

---

## Table of Contents

1. [Function Routing Matrix](#function-routing-matrix)
2. [Implementation Status Tracking](#implementation-status-tracking)
3. [Migration Strategy](#migration-strategy)
4. [Architecture Overview](#architecture-overview)

---

## Function Routing Matrix

### Legend
- ✅ **Compatible** - Function works natively on device
- ❌ **Incompatible** - Function not supported on device
- 🔄 **Translation** - Function needs translation/fallback
- ⚠️ **Limited** - Function works with limitations
- ❓ **Unknown** - Compatibility not yet determined

### Status Codes
- 🟢 **Complete** - Fully implemented and tested
- 🟡 **In Progress** - Implementation started
- 🔴 **Not Started** - Not yet implemented
- 🟣 **Testing** - Implementation complete, testing phase
- 🔵 **Design** - Design phase, not yet implemented

---


| 🟦 TORCH_CORE | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
|:-----------------|:---:|:----:|:---:|:---:|:------:|-----------|-------------|-------|
| `torch.AggregationType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: SUM AVG |
| `torch.AliasDb` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.AnyType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Argument` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ArgumentSpec` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.BFloat16Storage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.BenchmarkConfig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.BenchmarkExecutionStats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Block` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.BoolStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.BoolType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.BufferDict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ByteStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.CallStack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Capsule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.CharStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.ClassType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Code` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.CompilationUnit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.CompleteArgumentSpec` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ComplexDoubleStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.ComplexFloatStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.ComplexType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ConcreteModuleType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ConcreteModuleTypeBuilder` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.DeepCopyMemoTable` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.DeserializationStorageContext` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.DictType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.DisableTorchFunction` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.DisableTorchFunctionSubclass` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.DispatchKey` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: Undefined CompositeExplicitAutogradNonFunctional CompositeExplicitAutograd CompositeImplicitAutogradNestedTensor CompositeImplicitAutograd AutogradNestedTensor AutogradOther Autograd Conju... |
| `torch.DispatchKeySet` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.DoubleStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.EnumType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ErrorReport` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ExcludeDispatchKeyGuard` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ExecutionPlan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.FatalError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Common base class for all non-exit exceptions. |
| `torch.FileCheck` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.FloatStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.FloatType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.FunctionSchema` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Future` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.FutureType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Generator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Generator(device='cpu') -> Generator Creates and returns a generator object that manages the state of the algorithm which produces pseudo random numbers. Used as a keyword argument in many :ref:`in... |
| `torch.GradScaler` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device, init_scale, growth_factor, ...` | `None` | An instance ``scaler`` of :class:`GradScaler`. Helps perform the steps of gradient scaling conveniently. * ``scaler.scale(loss)`` multiplies a given loss by ``scaler``'s current scale factor. * ``s... |
| `torch.Gradient` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Graph` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.GraphExecutorState` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.HalfStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.IODescriptor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.InferredType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.IntStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.IntType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.InterfaceType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.JITException` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Common base class for all non-exit exceptions. |
| `torch.ListType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.LiteScriptModule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.LongStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.ModuleDict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Node` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.NoneType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.NumberType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.OperatorInfo` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.OptionalType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ParameterDict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.PyObjectType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.PyTorchFileReader` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.PyTorchFileWriter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.QInt32Storage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.QInt8Storage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.QUInt2x4Storage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.QUInt4x2Storage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.QUInt8Storage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.RRefType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ScriptClass` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ScriptClassFunction` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ScriptDict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ScriptDictIterator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ScriptDictKeyIterator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ScriptFunction` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Functionally equivalent to a :class:`ScriptModule`, but represents a single function and does not have any attributes or Parameters. |
| `torch.ScriptList` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ScriptListIterator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ScriptMethod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ScriptModule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ScriptModuleSerializer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ScriptObject` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ScriptObjectProperty` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.SerializationStorageContext` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ShortStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.Size` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.StaticModule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Storage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.StorageBase` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.StringType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.SymBool` | ❓ | ❓ | ❓ | ❓ | 🔴 | `node` | `Any` | Like a bool (including magic methods), but redirects all operations on the wrapped node. This is used in particular to symbolically record operations in the symbolic shape workflow. Unlike regular ... |
| `torch.SymBoolType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.SymFloat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `node` | `Any` | Like a float (including magic methods), but redirects all operations on the wrapped node. This is used in particular to symbolically record operations in the symbolic shape workflow. |
| `torch.SymInt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `node` | `Any` | Like an int (including magic methods), but redirects all operations on the wrapped node. This is used in particular to symbolically record operations in the symbolic shape workflow. |
| `torch.SymIntType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Tag` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: core cudagraph_unsafe data_dependent_output dynamic_output_shape flexible_layout generated inplace_view maybe_aliasing_or_mutating needs_exact_strides needs_fixed_stride_order nondetermini... |
| `torch.ThroughputBenchmark` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.TracingState` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.TupleType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.TypedStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.UnionType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.UntypedStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` |  |
| `torch.Use` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Value` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.add` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | add(input, other, *, alpha=1, out=None) -> Tensor Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`. .. math:: \text{{out}}_i = \text{{input}}_i + \text{{alpha}} \times \text{{other}}_i... |
| `torch.addbmm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | addbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None) -> Tensor Performs a batch matrix-matrix product of matrices stored in :attr:`batch1` and :attr:`batch2`, with a reduced add step (all ma... |
| `torch.addcdiv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | addcdiv(input, tensor1, tensor2, *, value=1, out=None) -> Tensor Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`, multiplies the result by the scalar :attr:`value` and adds... |
| `torch.addcmul` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | addcmul(input, tensor1, tensor2, *, value=1, out=None) -> Tensor Performs the element-wise multiplication of :attr:`tensor1` by :attr:`tensor2`, multiplies the result by the scalar :attr:`value` an... |
| `torch.addmm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | addmm(input, mat1, mat2, out_dtype=None, *, beta=1, alpha=1, out=None) -> Tensor Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`. The matrix :attr:`input` is added to... |
| `torch.addmv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | addmv(input, mat, vec, *, beta=1, alpha=1, out=None) -> Tensor Performs a matrix-vector product of the matrix :attr:`mat` and the vector :attr:`vec`. The vector :attr:`input` is added to the final ... |
| `torch.addmv_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.addr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | addr(input, vec1, vec2, *, beta=1, alpha=1, out=None) -> Tensor Performs the outer-product of vectors :attr:`vec1` and :attr:`vec2` and adds it to the matrix :attr:`input`. Optional values :attr:`b... |
| `torch.adjoint` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | adjoint(input: Tensor) -> Tensor Returns a view of the tensor conjugated and with the last two dimensions transposed. ``x.adjoint()`` is equivalent to ``x.transpose(-2, -1).conj()`` for complex ten... |
| `torch.affine_grid_generator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.alias_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.alias`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | all(input: Tensor) -> Tensor Tests if all elements in :attr:`input` evaluate to `True`. .. note:: This function matches the behaviour of NumPy in returning output of dtype `bool` for all supported ... |
| `torch.allclose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | allclose(input: Tensor, other: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool This function checks if :attr:`input` and :attr:`other` satisfy the condition: .. m... |
| `torch.alpha_dropout` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.alpha_dropout_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.amax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | amax(input, dim, keepdim=False, *, out=None) -> Tensor Returns the maximum value of each slice of the :attr:`input` tensor in the given dimension(s) :attr:`dim`. .. note:: The difference between ``... |
| `torch.amin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | amin(input, dim, keepdim=False, *, out=None) -> Tensor Returns the minimum value of each slice of the :attr:`input` tensor in the given dimension(s) :attr:`dim`. .. note:: The difference between ``... |
| `torch.aminmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | aminmax(input, *, dim=None, keepdim=False, out=None) -> (Tensor min, Tensor max) Computes the minimum and maximum values of the :attr:`input` tensor. Args: input (Tensor): The input tensor Keyword ... |
| `torch.angle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | angle(input: Tensor, *, out: Optional[Tensor]) -> Tensor Computes the element-wise angle (in radians) of the given :attr:`input` tensor. .. math:: \text{out}_{i} = angle(\text{input}_{i}) Args: inp... |
| `torch.any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | any(input: Tensor, *, out: Optional[Tensor]) -> Tensor Tests if any element in :attr:`input` evaluates to `True`. .. note:: This function matches the behaviour of NumPy in returning output of dtype... |
| `torch.are_deterministic_algorithms_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Returns True if the global deterministic flag is turned on. Refer to :func:`torch.use_deterministic_algorithms` documentation for more details. |
| `torch.argmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | argmax(input) -> LongTensor Returns the indices of the maximum value of all elements in the :attr:`input` tensor. This is the second value returned by :meth:`torch.max`. See its documentation for t... |
| `torch.argmin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | argmin(input, dim=None, keepdim=False) -> LongTensor Returns the indices of the minimum value(s) of the flattened tensor or along a dimension This is the second value returned by :meth:`torch.min`.... |
| `torch.argsort` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | argsort(input, dim=-1, descending=False, stable=False) -> Tensor Returns the indices that sort a tensor along a given dimension in ascending order by value. This is the second value returned by :me... |
| `torch.argwhere` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | argwhere(input) -> Tensor Returns a tensor containing the indices of all non-zero elements of :attr:`input`. Each row in the result contains the indices of a non-zero element in :attr:`input`. The ... |
| `torch.as_strided` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | as_strided(input, size, stride, storage_offset=None) -> Tensor Create a view of an existing `torch.Tensor` :attr:`input` with specified :attr:`size`, :attr:`stride` and :attr:`storage_offset`. .. w... |
| `torch.as_strided_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.as_strided_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.as_strided`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.as_strided_scatter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | as_strided_scatter(input, src, size, stride, storage_offset=None) -> Tensor Embeds the values of the :attr:`src` tensor into :attr:`input` along the elements corresponding to the result of calling ... |
| `torch.asarray` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | asarray(obj: Any, *, dtype: Optional[dtype], device: Optional[DeviceLikeType], copy: Optional[bool] = None, requires_grad: bool = False) -> Tensor # noqa: B950 Converts :attr:`obj` to a tensor. :at... |
| `torch.atleast_1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` | Returns a 1-dimensional view of each input tensor with zero dimensions. Input tensors with one or more dimensions are returned as-is. Args: input (Tensor or list of Tensors) Returns: output (Tensor... |
| `torch.atleast_2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` | Returns a 2-dimensional view of each input tensor with zero dimensions. Input tensors with two or more dimensions are returned as-is. Args: input (Tensor or list of Tensors) Returns: output (Tensor... |
| `torch.atleast_3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` | Returns a 3-dimensional view of each input tensor with zero dimensions. Input tensors with three or more dimensions are returned as-is. Args: input (Tensor or list of Tensors) Returns: output (Tens... |
| `torch.autocast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device_type, dtype, enabled, ...` | `Any` | Instances of :class:`autocast` serve as context managers or decorators that allow regions of your script to run in mixed precision. In these regions, ops run in an op-specific dtype chosen by autoc... |
| `torch.autocast_decrement_nesting` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.autocast_increment_nesting` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.baddbmm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | baddbmm(input, batch1, batch2, out_dtype=None, *, beta=1, alpha=1, out=None) -> Tensor Performs a batch matrix-matrix product of matrices in :attr:`batch1` and :attr:`batch2`. :attr:`input` is adde... |
| `torch.bartlett_window` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bartlett_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Bartlett window function. .. math:: w[n] = 1 - \left| \frac{2n}{N-1} -... |
| `torch.batch_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.batch_norm_backward_elemt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.batch_norm_backward_reduce` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.batch_norm_elemt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.batch_norm_gather_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.batch_norm_gather_stats_with_counts` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.batch_norm_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.batch_norm_update_stats` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.bincount` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bincount(input, weights=None, minlength=0) -> Tensor Count the frequency of each value in an array of non-negative ints. The number of bins (size 1) is one larger than the largest value in :attr:`i... |
| `torch.binomial` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.bitwise_and` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bitwise_and(input, other, *, out=None) -> Tensor Computes the bitwise AND of :attr:`input` and :attr:`other`. The input tensor must be of integral or Boolean types. For bool tensors, it computes th... |
| `torch.bitwise_left_shift` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bitwise_left_shift(input, other, *, out=None) -> Tensor Computes the left arithmetic shift of :attr:`input` by :attr:`other` bits. The input tensor must be of integral type. This operator supports ... |
| `torch.bitwise_not` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bitwise_not(input, *, out=None) -> Tensor Computes the bitwise NOT of the given input tensor. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical NOT. A... |
| `torch.bitwise_or` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bitwise_or(input: Tensor, other: Tensor, *, out: Optional[Tensor]) -> Tensor Computes the bitwise OR of :attr:`input` and :attr:`other`. The input tensor must be of integral or Boolean types. For b... |
| `torch.bitwise_right_shift` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bitwise_right_shift(input, other, *, out=None) -> Tensor Computes the right arithmetic shift of :attr:`input` by :attr:`other` bits. The input tensor must be of integral type. This operator support... |
| `torch.bitwise_xor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bitwise_xor(input, other, *, out=None) -> Tensor Computes the bitwise XOR of :attr:`input` and :attr:`other`. The input tensor must be of integral or Boolean types. For bool tensors, it computes th... |
| `torch.blackman_window` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | blackman_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Blackman window function. .. math:: w[n] = 0.42 - 0.5 \cos \left( \fra... |
| `torch.block_diag` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` | Create a block diagonal matrix from provided tensors. Args: *tensors: One or more tensors with 0, 1, or 2 dimensions. Returns: Tensor: A 2 dimensional tensor with all the input tensors arranged in ... |
| `torch.bmm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bmm(input, mat2, out_dtype=None, *, out=None) -> Tensor Performs a batch matrix-matrix product of matrices stored in :attr:`input` and :attr:`mat2`. :attr:`input` and :attr:`mat2` must be 3-D tenso... |
| `torch.broadcast_shapes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shapes` | `Any` | broadcast_shapes(*shapes) -> Size Similar to :func:`broadcast_tensors` but for shapes. This is equivalent to ``torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape`` but avoids the need crea... |
| `torch.broadcast_to` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | broadcast_to(input, shape) -> Tensor Broadcasts :attr:`input` to the shape :attr:`\shape`. Equivalent to calling ``input.expand(shape)``. See :meth:`~Tensor.expand` for details. Args: input (Tensor... |
| `torch.bucketize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bucketize(input, boundaries, *, out_int32=False, right=False, out=None) -> Tensor Returns the indices of the buckets to which each value in the :attr:`input` belongs, where the boundaries of the bu... |
| `torch.can_cast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | can_cast(from_, to) -> bool Determines if a type conversion is allowed under PyTorch casting rules described in the type promotion :ref:`documentation <type-promotion-doc>`. Args: from\_ (dtype): T... |
| `torch.cartesian_prod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `<class 'torch.Tensor'>` | Do cartesian product of the given sequence of tensors. The behavior is similar to python's `itertools.product`. Args: *tensors: any number of 1 dimensional tensors. Returns: Tensor: A tensor equiva... |
| `torch.cat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cat(tensors, dim=0, *, out=None) -> Tensor Concatenates the given sequence of tensors in :attr:`tensors` in the given dimension. All tensors must either have the same shape (except in the concatena... |
| `torch.ccol_indices_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cdist` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, p, ...` | `Any` | Computes batched the p-norm distance between each pair of the two collections of row vectors. Args: x1 (Tensor): input tensor where the last two dimensions represent the points and the feature dime... |
| `torch.celu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.celu_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | celu_(input, alpha=1.) -> Tensor In-place version of :func:`~celu`. |
| `torch.chain_matmul` | ❓ | ❓ | ❓ | ❓ | 🔴 | `matrices, out` | `Any` | Returns the matrix product of the :math:`N` 2-D tensors. This product is efficiently computed using the matrix chain order algorithm which selects the order in which incurs the lowest cost in terms... |
| `torch.channel_shuffle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | channel_shuffle(input, groups) -> Tensor Divide the channels in a tensor of shape :math:`(*, C , H, W)` into g groups and rearrange them as :math:`(*, C \frac g, g, H, W)`, while keeping the origin... |
| `torch.cholesky` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cholesky(input, upper=False, *, out=None) -> Tensor Computes the Cholesky decomposition of a symmetric positive-definite matrix :math:`A` or for batches of symmetric positive-definite matrices. If ... |
| `torch.cholesky_inverse` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cholesky_inverse(L, upper=False, *, out=None) -> Tensor Computes the inverse of a complex Hermitian or real symmetric positive-definite matrix given its Cholesky decomposition. Let :math:`A` be a c... |
| `torch.cholesky_solve` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cholesky_solve(B, L, upper=False, *, out=None) -> Tensor Computes the solution of a system of linear equations with complex Hermitian or real symmetric positive-definite lhs given its Cholesky deco... |
| `torch.choose_qparams_optimized` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.chunk` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | chunk(input: Tensor, chunks: int, dim: int = 0) -> Tuple[Tensor, ...] Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor. .. note:: This functi... |
| `torch.clamp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | clamp(input, min=None, max=None, *, out=None) -> Tensor Clamps all elements in :attr:`input` into the range `[` :attr:`min`, :attr:`max` `]`. Letting min_value and max_value be :attr:`min` and :att... |
| `torch.clamp_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.clamp_max` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.clamp_max_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.clamp_min` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.clamp_min_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.classproperty` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` |  |
| `torch.clear_autocast_cache` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.clip` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | clip(input, min=None, max=None, *, out=None) -> Tensor Alias for :func:`torch.clamp`. |
| `torch.clip_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.clone` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | clone(input, *, memory_format=torch.preserve_format) -> Tensor Returns a copy of :attr:`input`. .. note:: This function is differentiable, so gradients will flow back from the result of this operat... |
| `torch.col_indices_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.col_indices`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.column_stack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | column_stack(tensors, *, out=None) -> Tensor Creates a new tensor by horizontally stacking the tensors in :attr:`tensors`. Equivalent to ``torch.hstack(tensors)``, except each zero or one dimension... |
| `torch.combinations` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | combinations(input: Tensor, r: int = 2, with_replacement: bool = False) -> seq Compute combinations of length :math:`r` of the given tensor. The behavior is similar to python's `itertools.combinati... |
| `torch.compile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, fullgraph, dynamic, ...` | `typing.Union[typing.Callable[[typing.Callable[~_InputT, ~_RetT]], typing.Callable[~_InputT, ~_RetT]], typing.Callable[~_InputT, ~_RetT]]` | Optimizes given model/function using TorchDynamo and specified backend. If you are compiling an :class:`torch.nn.Module`, you can also use :meth:`torch.nn.Module.compile` to compile the module inpl... |
| `torch.compiled_with_cxx11_abi` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1 |
| `torch.complex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | complex(real, imag, *, out=None) -> Tensor Constructs a complex tensor with its real part equal to :attr:`real` and its imaginary part equal to :attr:`imag`. Args: real (Tensor): The real part of t... |
| `torch.concat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | concat(tensors, dim=0, *, out=None) -> Tensor Alias of :func:`torch.cat`. |
| `torch.concatenate` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | concatenate(tensors, axis=0, out=None) -> Tensor Alias of :func:`torch.cat`. |
| `torch.cond` | ❓ | ❓ | ❓ | ❓ | 🔴 | `pred, true_fn, false_fn, ...` | `typing.Any` | Conditionally applies `true_fn` or `false_fn`. .. warning:: `torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types and doesn't support training currently.... |
| `torch.conj` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conj(input) -> Tensor Returns a view of :attr:`input` with a flipped conjugate bit. If :attr:`input` has a non-complex dtype, this function just returns :attr:`input`. .. note:: :func:`torch.conj` ... |
| `torch.conj_physical` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conj_physical(input, *, out=None) -> Tensor Computes the element-wise conjugate of the given :attr:`input` tensor. If :attr:`input` has a non-complex dtype, this function just returns :attr:`input`... |
| `torch.conj_physical_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.copysign` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | copysign(input, other, *, out=None) -> Tensor Create a new floating-point tensor with the magnitude of :attr:`input` and the sign of :attr:`other`, elementwise. .. math:: \text{out}_{i} = \begin{ca... |
| `torch.corrcoef` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | corrcoef(input) -> Tensor Estimates the Pearson product-moment correlation coefficient matrix of the variables given by the :attr:`input` matrix, where rows are the variables and columns are the ob... |
| `torch.count_nonzero` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | count_nonzero(input, dim=None) -> Tensor Counts the number of non-zero values in the tensor :attr:`input` along the given :attr:`dim`. If no dim is specified then all non-zeros in the tensor are co... |
| `torch.cov` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cov(input, *, correction=1, fweights=None, aweights=None) -> Tensor Estimates the covariance matrix of the variables given by the :attr:`input` matrix, where rows are the variables and columns are ... |
| `torch.cross` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cross(input, other, dim=None, *, out=None) -> Tensor Returns the cross product of vectors in dimension :attr:`dim` of :attr:`input` and :attr:`other`. Supports input of float, double, cfloat and cd... |
| `torch.crow_indices_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.crow_indices`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.ctc_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cudnn_affine_grid_generator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cudnn_batch_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cudnn_grid_sampler` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cudnn_is_acceptable` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cummax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cummax(input, dim, *, out=None) -> (Tensor, LongTensor) Returns a namedtuple ``(values, indices)`` where ``values`` is the cumulative maximum of elements of :attr:`input` in the dimension :attr:`di... |
| `torch.cummin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cummin(input, dim, *, out=None) -> (Tensor, LongTensor) Returns a namedtuple ``(values, indices)`` where ``values`` is the cumulative minimum of elements of :attr:`input` in the dimension :attr:`di... |
| `torch.cumprod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cumprod(input, dim, *, dtype=None, out=None) -> Tensor Returns the cumulative product of elements of :attr:`input` in the dimension :attr:`dim`. For example, if :attr:`input` is a vector of size N,... |
| `torch.cumsum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cumsum(input, dim, *, dtype=None, out=None) -> Tensor Returns the cumulative sum of elements of :attr:`input` in the dimension :attr:`dim`. For example, if :attr:`input` is a vector of size N, the ... |
| `torch.cumulative_trapezoid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cumulative_trapezoid(y, x=None, *, dx=None, dim=-1) -> Tensor Cumulatively computes the `trapezoidal rule <https://en.wikipedia.org/wiki/Trapezoidal_rule>`_ along :attr:`dim`. By default the spacin... |
| `torch.deg2rad` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | deg2rad(input, *, out=None) -> Tensor Returns a new tensor with each of the elements of :attr:`input` converted from angles in degrees to radians. Args: input (Tensor): the input tensor. Keyword ar... |
| `torch.deg2rad_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.dequantize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | dequantize(tensor) -> Tensor Returns an fp32 Tensor by dequantizing a quantized Tensor Args: tensor (Tensor): A quantized Tensor .. function:: dequantize(tensors) -> sequence of Tensors :noindex: G... |
| `torch.det` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | det(input) -> Tensor Alias for :func:`torch.linalg.det` |
| `torch.detach` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.detach_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.detach_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.detach`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.diag` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | diag(input, diagonal=0, *, out=None) -> Tensor - If :attr:`input` is a vector (1-D tensor), then returns a 2-D square tensor with the elements of :attr:`input` as the diagonal. - If :attr:`input` i... |
| `torch.diag_embed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | diag_embed(input, offset=0, dim1=-2, dim2=-1) -> Tensor Creates a tensor whose diagonals of certain 2D planes (specified by :attr:`dim1` and :attr:`dim2`) are filled by :attr:`input`. To facilitate... |
| `torch.diagflat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | diagflat(input, offset=0) -> Tensor - If :attr:`input` is a vector (1-D tensor), then returns a 2-D square tensor with the elements of :attr:`input` as the diagonal. - If :attr:`input` is a tensor ... |
| `torch.diagonal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | diagonal(input, offset=0, dim1=0, dim2=1) -> Tensor Returns a partial view of :attr:`input` with the its diagonal elements with respect to :attr:`dim1` and :attr:`dim2` appended as a dimension at t... |
| `torch.diagonal_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.diagonal`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.diagonal_scatter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | diagonal_scatter(input, src, offset=0, dim1=0, dim2=1) -> Tensor Embeds the values of the :attr:`src` tensor into :attr:`input` along the diagonal elements of :attr:`input`, with respect to :attr:`... |
| `torch.diff` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | diff(input, n=1, dim=-1, prepend=None, append=None) -> Tensor Computes the n-th forward difference along the given dimension. The first-order differences are given by `out[i] = input[i + 1] - input... |
| `torch.digamma` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | digamma(input, *, out=None) -> Tensor Alias for :func:`torch.special.digamma`. |
| `torch.dist` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | dist(input, other, p=2) -> Tensor Returns the p-norm of (:attr:`input` - :attr:`other`) The shapes of :attr:`input` and :attr:`other` must be :ref:`broadcastable <broadcasting-semantics>`. Args: in... |
| `torch.div` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | div(input, other, *, rounding_mode=None, out=None) -> Tensor Divides each element of the input ``input`` by the corresponding element of :attr:`other`. .. math:: \text{out}_i = \frac{\text{input}_i... |
| `torch.divide` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | divide(input, other, *, rounding_mode=None, out=None) -> Tensor Alias for :func:`torch.div`. |
| `torch.dot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | dot(input, tensor, *, out=None) -> Tensor Computes the dot product of two 1D tensors. .. note:: Unlike NumPy's dot, torch.dot intentionally only supports computing the dot product of two 1D tensors... |
| `torch.dropout` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.dropout_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.dsmm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.dsplit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | dsplit(input, indices_or_sections) -> List of Tensors Splits :attr:`input`, a tensor with three or more dimensions, into multiple tensors depthwise according to :attr:`indices_or_sections`. Each sp... |
| `torch.dstack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | dstack(tensors, *, out=None) -> Tensor Stack tensors in sequence depthwise (along third axis). This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been reshaped ... |
| `torch.dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.eig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `self, eigenvectors, e, ...` | `tuple[torch.Tensor, torch.Tensor]` |  |
| `torch.einsum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args` | `<class 'torch.Tensor'>` | einsum(equation, *operands) -> Tensor Sums the product of the elements of the input :attr:`operands` along dimensions specified using a notation based on the Einstein summation convention. Einsum a... |
| `torch.embedding` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.embedding_bag` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.embedding_renorm_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.enable_grad` | ❓ | ❓ | ❓ | ❓ | 🔴 | `orig_func` | `Any` | Context-manager that enables gradient calculation. Enables gradient calculation, if it has been disabled via :class:`~no_grad` or :class:`~set_grad_enabled`. This context manager is thread local; i... |
| `torch.eq` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | eq(input, other, *, out=None) -> Tensor Computes element-wise equality The second argument can be a number or a tensor whose shape is :ref:`broadcastable <broadcasting-semantics>` with the first ar... |
| `torch.equal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | equal(input, other) -> bool ``True`` if two tensors have the same size and elements, ``False`` otherwise. .. note:: Tensors containing NaNs are never equal to each other. Additionally, this functio... |
| `torch.erf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | erf(input, *, out=None) -> Tensor Alias for :func:`torch.special.erf`. |
| `torch.erf_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.erfc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | erfc(input, *, out=None) -> Tensor Alias for :func:`torch.special.erfc`. |
| `torch.erfc_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.erfinv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | erfinv(input, *, out=None) -> Tensor Alias for :func:`torch.special.erfinv`. |
| `torch.fake_quantize_per_channel_affine` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | fake_quantize_per_channel_affine(input, scale, zero_point, axis, quant_min, quant_max) -> Tensor Returns a new tensor with the data in :attr:`input` fake quantized per channel using :attr:`scale`, ... |
| `torch.fbgemm_pack_gemm_matrix_fp16` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.fbgemm_pack_quantized_matrix` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.feature_alpha_dropout` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.feature_alpha_dropout_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.feature_dropout` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.feature_dropout_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.fill` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.fill_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.finfo` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.fix` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | fix(input, *, out=None) -> Tensor Alias for :func:`torch.trunc` |
| `torch.fix_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.flatten` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | flatten(input, start_dim=0, end_dim=-1) -> Tensor Flattens :attr:`input` by reshaping it into a one-dimensional tensor. If :attr:`start_dim` or :attr:`end_dim` are passed, only dimensions starting ... |
| `torch.flip` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | flip(input, dims) -> Tensor Reverse the order of an n-D tensor along given axis in dims. .. note:: `torch.flip` makes a copy of :attr:`input`'s data. This is different from NumPy's `np.flip`, which... |
| `torch.fliplr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | fliplr(input) -> Tensor Flip tensor in the left/right direction, returning a new tensor. Flip the entries in each row in the left/right direction. Columns are preserved, but appear in a different o... |
| `torch.flipud` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | flipud(input) -> Tensor Flip tensor in the up/down direction, returning a new tensor. Flip the entries in each column in the up/down direction. Rows are preserved, but appear in a different order t... |
| `torch.float_power` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | float_power(input, exponent, *, out=None) -> Tensor Raises :attr:`input` to the power of :attr:`exponent`, elementwise, in double precision. If neither input is complex returns a ``torch.float64`` ... |
| `torch.fmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | fmax(input, other, *, out=None) -> Tensor Computes the element-wise maximum of :attr:`input` and :attr:`other`. This is like :func:`torch.maximum` except it handles NaNs differently: if exactly one... |
| `torch.fmin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | fmin(input, other, *, out=None) -> Tensor Computes the element-wise minimum of :attr:`input` and :attr:`other`. This is like :func:`torch.minimum` except it handles NaNs differently: if exactly one... |
| `torch.fmod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | fmod(input, other, *, out=None) -> Tensor Applies C++'s `std::fmod <https://en.cppreference.com/w/cpp/numeric/math/fmod>`_ entrywise. The result has the same sign as the dividend :attr:`input` and ... |
| `torch.fork` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | fork(*args, **kwargs) -> torch._C.Future |
| `torch.frac` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | frac(input, *, out=None) -> Tensor Computes the fractional portion of each element in :attr:`input`. .. math:: \text{out}_{i} = \text{input}_{i} - \left\lfloor |\text{input}_{i}| \right\rfloor * \o... |
| `torch.frac_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.frobenius_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.from_dlpack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ext_tensor` | `torch.Tensor` | from_dlpack(ext_tensor) -> Tensor Converts a tensor from an external library into a ``torch.Tensor``. The returned PyTorch tensor will share the memory with the input tensor (which may have come fr... |
| `torch.from_file` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | from_file(filename, shared=None, size=0, *, dtype=None, layout=None, device=None, pin_memory=False) Creates a CPU tensor with a storage backed by a memory-mapped file. If ``shared`` is True, then m... |
| `torch.from_numpy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | from_numpy(ndarray) -> Tensor Creates a :class:`Tensor` from a :class:`numpy.ndarray`. The returned tensor and :attr:`ndarray` share the same memory. Modifications to the tensor will be reflected i... |
| `torch.frombuffer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False) -> Tensor Creates a 1-dimensional :class:`Tensor` from an object that implements the Python buffer protocol. Skips the first :a... |
| `torch.full` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Creates a tensor of size :attr:`size` filled with :attr:`fill_value`. The tensor's ... |
| `torch.full_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | full_like(input, fill_value, \*, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor Returns a tensor with the same size as :attr:`inp... |
| `torch.fused_moving_avg_obs_fake_quant` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.gather` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | gather(input, dim, index, *, sparse_grad=False, out=None) -> Tensor Gathers values along an axis specified by `dim`. For a 3-D tensor the output is specified by:: out[i][j][k] = input[index[i][j][k... |
| `torch.gcd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | gcd(input, other, *, out=None) -> Tensor Computes the element-wise greatest common divisor (GCD) of :attr:`input` and :attr:`other`. Both :attr:`input` and :attr:`other` must have integer types. ..... |
| `torch.gcd_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ge` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ge(input, other, *, out=None) -> Tensor Computes :math:`\text{input} \geq \text{other}` element-wise. The second argument can be a number or a tensor whose shape is :ref:`broadcastable <broadcastin... |
| `torch.geqrf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | geqrf(input, *, out=None) -> (Tensor, Tensor) This is a low-level function for calling LAPACK's geqrf directly. This function returns a namedtuple (a, tau) as defined in `LAPACK documentation for g... |
| `torch.ger` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ger(input, vec2, *, out=None) -> Tensor Alias of :func:`torch.outer`. .. warning:: This function is deprecated and will be removed in a future PyTorch release. Use :func:`torch.outer` instead. |
| `torch.get_autocast_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.get_autocast_gpu_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.get_autocast_ipu_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.get_autocast_xla_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.get_default_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | get_default_dtype() -> torch.dtype Get the current default floating point :class:`torch.dtype`. Example:: >>> torch.get_default_dtype() # initial default for floating point is torch.float32 torch.f... |
| `torch.get_deterministic_debug_mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Returns the current value of the debug mode for deterministic operations. Refer to :func:`torch.set_deterministic_debug_mode` documentation for more details. |
| `torch.get_file_path` | ❓ | ❓ | ❓ | ❓ | 🔴 | `path_components` | `<class 'str'>` |  |
| `torch.get_float32_matmul_precision` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'str'>` | Returns the current value of float32 matrix multiplication precision. Refer to :func:`torch.set_float32_matmul_precision` documentation for more details. |
| `torch.get_num_interop_threads` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | get_num_interop_threads() -> int Returns the number of threads used for inter-op parallelism on CPU (e.g. in JIT interpreter) |
| `torch.get_num_threads` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | get_num_threads() -> int Returns the number of threads used for parallelizing CPU operations |
| `torch.get_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'torch.Tensor'>` | Returns the random number generator state as a `torch.ByteTensor`. .. note:: The returned state is for the default generator on CPU only. See also: :func:`torch.random.fork_rng`. |
| `torch.gradient` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | gradient(input, *, spacing=1, dim=None, edge_order=1) -> List of Tensors Estimates the gradient of a function :math:`g : \mathbb{R}^n \rightarrow \mathbb{R}` in one or more dimensions using the `se... |
| `torch.greater` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | greater(input, other, *, out=None) -> Tensor Alias for :func:`torch.gt`. |
| `torch.greater_equal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | greater_equal(input, other, *, out=None) -> Tensor Alias for :func:`torch.ge`. |
| `torch.grid_sampler` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.grid_sampler_2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.grid_sampler_3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.group_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.gru` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.gru_cell` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.gt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | gt(input, other, *, out=None) -> Tensor Computes :math:`\text{input} > \text{other}` element-wise. The second argument can be a number or a tensor whose shape is :ref:`broadcastable <broadcasting-s... |
| `torch.hamming_window` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Hamming window function. .. math:: w[n] = \alpha... |
| `torch.hann_window` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hann_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Hann window function. .. math:: w[n] = \frac{1}{2}\ \left[1 - \cos \left( ... |
| `torch.hardshrink` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hardshrink(input, lambd=0.5) -> Tensor Applies the hard shrinkage function element-wise See :class:`~torch.nn.Hardshrink` for more details. |
| `torch.heaviside` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | heaviside(input, values, *, out=None) -> Tensor Computes the Heaviside step function for each element in :attr:`input`. The Heaviside step function is defined as: .. math:: \text{{heaviside}}(input... |
| `torch.hinge_embedding_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.histc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | histc(input, bins=100, min=0, max=0, *, out=None) -> Tensor Computes the histogram of a tensor. The elements are sorted into equal width bins between :attr:`min` and :attr:`max`. If :attr:`min` and... |
| `torch.histogram` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | histogram(input, bins, *, range=None, weight=None, density=False, out=None) -> (Tensor, Tensor) Computes a histogram of the values in a tensor. :attr:`bins` can be an integer or a 1D tensor. If :at... |
| `torch.histogramdd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | histogramdd(input, bins, *, range=None, weight=None, density=False, out=None) -> (Tensor, Tensor[]) Computes a multi-dimensional histogram of the values in a tensor. Interprets the elements of an i... |
| `torch.hsmm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.hsplit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hsplit(input, indices_or_sections) -> List of Tensors Splits :attr:`input`, a tensor with one or more dimensions, into multiple tensors horizontally according to :attr:`indices_or_sections`. Each s... |
| `torch.hspmm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hspmm(mat1, mat2, *, out=None) -> Tensor Performs a matrix multiplication of a :ref:`sparse COO matrix <sparse-coo-docs>` :attr:`mat1` and a strided matrix :attr:`mat2`. The result is a (1 + 1)-dim... |
| `torch.hstack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hstack(tensors, *, out=None) -> Tensor Stack tensors in sequence horizontally (column wise). This is equivalent to concatenation along the first axis for 1-D tensors, and along the second axis for ... |
| `torch.hypot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hypot(input, other, *, out=None) -> Tensor Given the legs of a right triangle, return its hypotenuse. .. math:: \text{out}_{i} = \sqrt{\text{input}_{i}^{2} + \text{other}_{i}^{2}} The shapes of ``i... |
| `torch.i0` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | i0(input, *, out=None) -> Tensor Alias for :func:`torch.special.i0`. |
| `torch.i0_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.igamma` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | igamma(input, other, *, out=None) -> Tensor Alias for :func:`torch.special.gammainc`. |
| `torch.igammac` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | igammac(input, other, *, out=None) -> Tensor Alias for :func:`torch.special.gammaincc`. |
| `torch.iinfo` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.imag` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | imag(input) -> Tensor Returns a new tensor containing imaginary values of the :attr:`self` tensor. The returned tensor and :attr:`self` share the same underlying storage. .. warning:: :func:`imag` ... |
| `torch.import_ir_module` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | import_ir_module(arg0: torch._C.CompilationUnit, arg1: str, arg2: object, arg3: dict, arg4: bool) -> torch._C.ScriptModule |
| `torch.import_ir_module_from_buffer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | import_ir_module_from_buffer(arg0: torch._C.CompilationUnit, arg1: str, arg2: object, arg3: dict, arg4: bool) -> torch._C.ScriptModule |
| `torch.index_add` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | index_add(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[Number, _complex] = 1, out: Optional[Tensor]) -> Tensor # noqa: B950 See :meth:`~Tensor.index_add_` for function de... |
| `torch.index_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | index_copy(input: Tensor, dim: int, index: Tensor, source: Tensor, *, out: Optional[Tensor]) -> Tensor See :meth:`~Tensor.index_add_` for function description. |
| `torch.index_fill` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.index_put` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.index_put_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.index_reduce` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | index_reduce(input: Tensor, dim: int, index: Tensor, source: Tensor, reduce: str, *, include_self: bool = True, out: Optional[Tensor]) -> Tensor # noqa: B950 See :meth:`~Tensor.index_reduce_` for f... |
| `torch.index_select` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | index_select(input, dim, index, *, out=None) -> Tensor Returns a new tensor which indexes the :attr:`input` tensor along dimension :attr:`dim` using the entries in :attr:`index` which is a `LongTen... |
| `torch.indices_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.indices`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.inference_mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mode` | `Any` | Context-manager that enables or disables inference mode. InferenceMode is a context manager analogous to :class:`~no_grad` to be used when you are certain your operations will have no interactions ... |
| `torch.init_num_threads` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | init_num_threads() -> None init_num_threads() Initializes the number of parallel threads used on the current thread. Call this whenever a new thread is created in order to propagate values from :fu... |
| `torch.initial_seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Returns the initial seed for generating random numbers as a Python `long`. .. note:: The returned seed is for the default generator on CPU only. |
| `torch.inner` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | inner(input, other, *, out=None) -> Tensor Computes the dot product for 1D tensors. For higher dimensions, sums the product of elements from :attr:`input` and :attr:`other` along their last dimensi... |
| `torch.int_repr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.inverse` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | inverse(input, *, out=None) -> Tensor Alias for :func:`torch.linalg.inv` |
| `torch.is_anomaly_check_nan_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.is_anomaly_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.is_autocast_cache_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.is_autocast_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.is_autocast_ipu_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.is_autocast_xla_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.is_complex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | is_complex(input) -> (bool) Returns True if the data type of :attr:`input` is a complex data type i.e., one of ``torch.complex64``, and ``torch.complex128``. Args: input (Tensor): the input tensor. |
| `torch.is_conj` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | is_conj(input) -> (bool) Returns True if the :attr:`input` is a conjugated tensor, i.e. its conjugate bit is set to `True`. Args: input (Tensor): the input tensor. |
| `torch.is_deterministic_algorithms_warn_only_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Returns True if the global deterministic flag is set to warn only. Refer to :func:`torch.use_deterministic_algorithms` documentation for more details. |
| `torch.is_distributed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.is_floating_point` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | is_floating_point(input) -> (bool) Returns True if the data type of :attr:`input` is a floating point data type i.e., one of ``torch.float64``, ``torch.float32``, ``torch.float16``, and ``torch.bfl... |
| `torch.is_grad_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | is_grad_enabled() -> (bool) Returns True if grad mode is currently enabled. |
| `torch.is_inference` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | is_inference(input) -> (bool) Returns True if :attr:`input` is an inference tensor. A non-view tensor is an inference tensor if and only if it was allocated during inference mode. A view tensor is ... |
| `torch.is_inference_mode_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | is_inference_mode_enabled() -> (bool) Returns True if inference mode is currently enabled. |
| `torch.is_neg` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.is_nonzero` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | is_nonzero(input) -> (bool) Returns True if the :attr:`input` is a single element tensor which is not equal to zero after type conversions. i.e. not equal to ``torch.tensor([0.])`` or ``torch.tenso... |
| `torch.is_same_size` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.is_signed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.is_storage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj` | `typing_extensions.TypeIs[typing.Union[ForwardRef('TypedStorage'), ForwardRef('UntypedStorage')]]` | Returns True if `obj` is a PyTorch storage object. Args: obj (Object): Object to test |
| `torch.is_vulkan_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.is_warn_always_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Returns True if the global warn_always flag is turned on. Refer to :func:`torch.set_warn_always` documentation for more details. |
| `torch.isclose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor Returns a new tensor with boolean elements representing if each element of :attr:`input` is "close" to the corresponding ele... |
| `torch.isfinite` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | isfinite(input) -> Tensor Returns a new tensor with boolean elements representing if each element is `finite` or not. Real values are finite when they are not NaN, negative infinity, or infinity. C... |
| `torch.isnan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | isnan(input) -> Tensor Returns a new tensor with boolean elements representing if each element of :attr:`input` is NaN or not. Complex values are considered NaN when either their real and/or imagin... |
| `torch.isneginf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | isneginf(input, *, out=None) -> Tensor Tests if each element of :attr:`input` is negative infinity or not. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output t... |
| `torch.isreal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | isreal(input) -> Tensor Returns a new tensor with boolean elements representing if each element of :attr:`input` is real-valued or not. All real-valued types are considered real. Complex values are... |
| `torch.istft` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | istft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False, onesided=None, length=None, return_complex=False) -> Tensor: Inverse short time Fourier Transform. ... |
| `torch.kaiser_window` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | kaiser_window(window_length, periodic=True, beta=12.0, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Computes the Kaiser window with window length :attr:`window_l... |
| `torch.kl_div` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.kron` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | kron(input, other, *, out=None) -> Tensor Computes the Kronecker product, denoted by :math:`\otimes`, of :attr:`input` and :attr:`other`. If :attr:`input` is a :math:`(a_0 \times a_1 \times \dots \... |
| `torch.kthvalue` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | kthvalue(input, k, dim=None, keepdim=False, *, out=None) -> (Tensor, LongTensor) Returns a namedtuple ``(values, indices)`` where ``values`` is the :attr:`k` th smallest element of each row of the ... |
| `torch.layer_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.layout` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.lcm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | lcm(input, other, *, out=None) -> Tensor Computes the element-wise least common multiple (LCM) of :attr:`input` and :attr:`other`. Both :attr:`input` and :attr:`other` must have integer types. .. n... |
| `torch.lcm_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.le` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | le(input, other, *, out=None) -> Tensor Computes :math:`\text{input} \leq \text{other}` element-wise. The second argument can be a number or a tensor whose shape is :ref:`broadcastable <broadcastin... |
| `torch.lerp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | lerp(input, end, weight, *, out=None) Does a linear interpolation of two tensors :attr:`start` (given by :attr:`input`) and :attr:`end` based on a scalar or tensor :attr:`weight` and returns the re... |
| `torch.less` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | less(input, other, *, out=None) -> Tensor Alias for :func:`torch.lt`. |
| `torch.less_equal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | less_equal(input, other, *, out=None) -> Tensor Alias for :func:`torch.le`. |
| `torch.lgamma` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | lgamma(input, *, out=None) -> Tensor Computes the natural logarithm of the absolute value of the gamma function on :attr:`input`. .. math:: \text{out}_{i} = \ln |\Gamma(\text{input}_{i})| Args: inp... |
| `torch.load` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f, map_location, pickle_module, ...` | `typing.Any` | load(f, map_location=None, pickle_module=pickle, *, weights_only=True, mmap=None, **pickle_load_args) Loads an object saved with :func:`torch.save` from a file. :func:`torch.load` uses Python's unp... |
| `torch.lobpcg` | ❓ | ❓ | ❓ | ❓ | 🔴 | `A, k, B, ...` | `tuple[torch.Tensor, torch.Tensor]` | Find the k largest (or smallest) eigenvalues and the corresponding eigenvectors of a symmetric positive definite generalized eigenvalue problem using matrix-free LOBPCG methods. This function is a ... |
| `torch.lstm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.lstm_cell` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.lstsq` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, A, out` | `tuple[torch.Tensor, torch.Tensor]` |  |
| `torch.lt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | lt(input, other, *, out=None) -> Tensor Computes :math:`\text{input} < \text{other}` element-wise. The second argument can be a number or a tensor whose shape is :ref:`broadcastable <broadcasting-s... |
| `torch.lu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Computes the LU factorization of a matrix or batches of matrices :attr:`A`. Returns a tuple containing the LU factorization and pivots of :attr:`A`. Pivoting is done if :attr:`pivot` is set to ``Tr... |
| `torch.lu_solve` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | lu_solve(b, LU_data, LU_pivots, *, out=None) -> Tensor Returns the LU solve of the linear system :math:`Ax = b` using the partially pivoted LU factorization of A from :func:`~linalg.lu_factor`. Thi... |
| `torch.lu_unpack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True, *, out=None) -> (Tensor, Tensor, Tensor) Unpacks the LU decomposition returned by :func:`~linalg.lu_factor` into the `P, L, U` ma... |
| `torch.manual_seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `seed` | `<class 'torch._C.Generator'>` | Sets the seed for generating random numbers on all devices. Returns a `torch.Generator` object. Args: seed (int): The desired seed. Value must be within the inclusive range `[-0x8000_0000_0000_0000... |
| `torch.margin_ranking_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.masked_fill` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.masked_scatter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.masked_select` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | masked_select(input, mask, *, out=None) -> Tensor Returns a new 1-D tensor which indexes the :attr:`input` tensor according to the boolean mask :attr:`mask` which is a `BoolTensor`. The shapes of t... |
| `torch.matmul` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | matmul(input, other, *, out=None) -> Tensor Matrix product of two tensors. The behavior depends on the dimensionality of the tensors as follows: - If both tensors are 1-dimensional, the dot product... |
| `torch.matrix_power` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | matrix_power(input, n, *, out=None) -> Tensor Alias for :func:`torch.linalg.matrix_power` |
| `torch.matrix_rank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, tol, symmetric, ...` | `<class 'torch.Tensor'>` |  |
| `torch.max` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | max(input) -> Tensor Returns the maximum value of all elements in the ``input`` tensor. Args: input (Tensor): the input tensor. Example:: >>> a = torch.randn(1, 3) >>> a tensor([[ 0.6763, 0.7445, -... |
| `torch.maximum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | maximum(input, other, *, out=None) -> Tensor Computes the element-wise maximum of :attr:`input` and :attr:`other`. .. note:: If one of the elements being compared is a NaN, then that element is ret... |
| `torch.mean` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | mean(input, *, dtype=None) -> Tensor .. note:: If the `input` tensor is empty, ``torch.mean()`` returns ``nan``. This behavior is consistent with NumPy and follows the definition that the mean over... |
| `torch.median` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | median(input) -> Tensor Returns the median of the values in :attr:`input`. .. note:: The median is not unique for :attr:`input` tensors with an even number of elements. In this case the lower of th... |
| `torch.merge_type_from_type_comment` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | merge_type_from_type_comment(arg0: torch._C._jit_tree_views.Decl, arg1: torch._C._jit_tree_views.Decl, arg2: bool) -> torch._C._jit_tree_views.Decl |
| `torch.meshgrid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, indexing` | `tuple[torch.Tensor, ...]` | Creates grids of coordinates specified by the 1D inputs in `attr`:tensors. This is helpful when you want to visualize data over some range of inputs. See below for a plotting example. Given :math:`... |
| `torch.min` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | min(input) -> Tensor Returns the minimum value of all elements in the :attr:`input` tensor. Args: input (Tensor): the input tensor. Example:: >>> a = torch.randn(1, 3) >>> a tensor([[ 0.6750, 1.085... |
| `torch.minimum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | minimum(input, other, *, out=None) -> Tensor Computes the element-wise minimum of :attr:`input` and :attr:`other`. .. note:: If one of the elements being compared is a NaN, then that element is ret... |
| `torch.miopen_batch_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.miopen_rnn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.mkldnn_rnn_layer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.mm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | mm(input, mat2, out_dtype=None, *, out=None) -> Tensor Performs a matrix multiplication of the matrices :attr:`input` and :attr:`mat2`. If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`mat... |
| `torch.mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | mode(input, dim=-1, keepdim=False, *, out=None) -> (Tensor, LongTensor) Returns a namedtuple ``(values, indices)`` where ``values`` is the mode value of each row of the :attr:`input` tensor in the ... |
| `torch.moveaxis` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | moveaxis(input, source, destination) -> Tensor Alias for :func:`torch.movedim`. This function is equivalent to NumPy's moveaxis function. Examples:: >>> t = torch.randn(3,2,1) >>> t tensor([[[-0.33... |
| `torch.movedim` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | movedim(input, source, destination) -> Tensor Moves the dimension(s) of :attr:`input` at the position(s) in :attr:`source` to the position(s) in :attr:`destination`. Other dimensions of :attr:`inpu... |
| `torch.msort` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | msort(input: Tensor, *, out: Optional[Tensor]) -> Tensor Sorts the elements of the :attr:`input` tensor along its first dimension in ascending order by value. .. note:: `torch.msort(t)` is equivale... |
| `torch.mul` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | mul(input, other, *, out=None) -> Tensor Multiplies :attr:`input` by :attr:`other`. .. math:: \text{out}_i = \text{input}_i \times \text{other}_i Supports :ref:`broadcasting to a common shape <broa... |
| `torch.multinomial` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | multinomial(input, num_samples, replacement=False, *, generator=None, out=None) -> LongTensor Returns a tensor where each row contains :attr:`num_samples` indices sampled from the multinomial (a st... |
| `torch.multiply` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | multiply(input, other, *, out=None) Alias for :func:`torch.mul`. |
| `torch.mv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | mv(input, vec, *, out=None) -> Tensor Performs a matrix-vector product of the matrix :attr:`input` and the vector :attr:`vec`. If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-... |
| `torch.mvlgamma` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | mvlgamma(input, p, *, out=None) -> Tensor Alias for :func:`torch.special.multigammaln`. |
| `torch.nan_to_num` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None) -> Tensor Replaces :literal:`NaN`, positive infinity, and negative infinity values in :attr:`input` with the values specified by :a... |
| `torch.nan_to_num_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.nanmean` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | nanmean(input, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor Computes the mean of all `non-NaN` elements along the specified dimensions. Input must be floating point or complex. This ... |
| `torch.nanmedian` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | nanmedian(input) -> Tensor Returns the median of the values in :attr:`input`, ignoring ``NaN`` values. This function is identical to :func:`torch.median` when there are no ``NaN`` values in :attr:`... |
| `torch.nanquantile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | nanquantile(input, q, dim=None, keepdim=False, *, interpolation='linear', out=None) -> Tensor This is a variant of :func:`torch.quantile` that "ignores" ``NaN`` values, computing the quantiles :att... |
| `torch.nansum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | nansum(input, *, dtype=None) -> Tensor Returns the sum of all elements, treating Not a Numbers (NaNs) as zero. Args: input (Tensor): the input tensor. Keyword args: dtype (:class:`torch.dtype`, opt... |
| `torch.narrow` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | narrow(input, dim, start, length) -> Tensor Returns a new tensor that is a narrowed version of :attr:`input` tensor. The dimension :attr:`dim` is input from :attr:`start` to ``start + length``. The... |
| `torch.narrow_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | narrow_copy(input, dim, start, length, *, out=None) -> Tensor Same as :meth:`Tensor.narrow` except this returns a copy rather than shared storage. This is primarily for sparse tensors, which do not... |
| `torch.native_batch_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.native_channel_shuffle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | native_channel_shuffle(input, groups) -> Tensor Native kernel level implementation of the `channel_shuffle`. This function might become private in future releases, use with caution. Divide the chan... |
| `torch.native_dropout` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.native_group_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.native_layer_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.native_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ne` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ne(input, other, *, out=None) -> Tensor Computes :math:`\text{input} \neq \text{other}` element-wise. The second argument can be a number or a tensor whose shape is :ref:`broadcastable <broadcastin... |
| `torch.neg` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | neg(input, *, out=None) -> Tensor Returns a new tensor with the negative of the elements of :attr:`input`. .. math:: \text{out} = -1 \times \text{input} Args: input (Tensor): the input tensor. Keyw... |
| `torch.neg_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.negative` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | negative(input, *, out=None) -> Tensor Alias for :func:`torch.neg` |
| `torch.negative_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.nextafter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | nextafter(input, other, *, out=None) -> Tensor Return the next floating-point value after :attr:`input` towards :attr:`other`, elementwise. The shapes of ``input`` and ``other`` must be :ref:`broad... |
| `torch.no_grad` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Context-manager that disables gradient calculation. Disabling gradient calculation is useful for inference, when you are sure that you will not call :meth:`Tensor.backward()`. It will reduce memory... |
| `torch.nonzero` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | nonzero(input, *, out=None, as_tuple=False) -> LongTensor or tuple of LongTensors .. note:: :func:`torch.nonzero(..., as_tuple=False) <torch.nonzero>` (default) returns a 2-D tensor where each row ... |
| `torch.nonzero_static` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, p, dim, ...` | `Any` | Returns the matrix norm or vector norm of a given tensor. .. warning:: torch.norm is deprecated and may be removed in a future PyTorch release. Its documentation and behavior may be incorrect, and ... |
| `torch.norm_except_dim` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.not_equal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | not_equal(input, other, *, out=None) -> Tensor Alias for :func:`torch.ne`. |
| `torch.nuclear_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.numel` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | numel(input: Tensor) -> int Returns the total number of elements in the :attr:`input` tensor. Args: input (Tensor): the input tensor. Example:: >>> a = torch.randn(1, 2, 3, 4, 5) >>> torch.numel(a)... |
| `torch.orgqr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | orgqr(input, tau) -> Tensor Alias for :func:`torch.linalg.householder_product`. |
| `torch.ormqr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ormqr(input, tau, other, left=True, transpose=False, *, out=None) -> Tensor Computes the matrix-matrix multiplication of a product of Householder matrices with a general matrix. Multiplies a :math:... |
| `torch.outer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | outer(input, vec2, *, out=None) -> Tensor Outer product of :attr:`input` and :attr:`vec2`. If :attr:`input` is a vector of size :math:`n` and :attr:`vec2` is a vector of size :math:`m`, then :attr:... |
| `torch.parse_ir` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | parse_ir(input: str, parse_tensor_constants: bool = False) -> torch::jit::Graph |
| `torch.parse_schema` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | parse_schema(schema: str, allow_typevars: bool = True) -> c10::FunctionSchema |
| `torch.parse_type_comment` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | parse_type_comment(arg0: str) -> torch._C._jit_tree_views.Decl |
| `torch.pca_lowrank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `A, q, center, ...` | `tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | Performs linear Principal Component Analysis (PCA) on a low-rank matrix, batches of such matrices, or sparse matrix. This function returns a namedtuple ``(U, S, V)`` which is the nearly optimal app... |
| `torch.pdist` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | pdist(input, p=2) -> Tensor Computes the p-norm distance between every pair of row vectors in the input. This is identical to the upper triangular portion, excluding the diagonal, of `torch.norm(in... |
| `torch.permute` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | permute(input, dims) -> Tensor Returns a view of the original tensor :attr:`input` with its dimensions permuted. Args: input (Tensor): the input tensor. dims (tuple of int): The desired ordering of... |
| `torch.permute_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.permute`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.pinverse` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | pinverse(input, rcond=1e-15) -> Tensor Alias for :func:`torch.linalg.pinv` |
| `torch.pixel_shuffle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | pixel_shuffle(input, upscale_factor) -> Tensor Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)` to a tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is the :... |
| `torch.pixel_unshuffle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | pixel_unshuffle(input, downscale_factor) -> Tensor Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements in a tensor of shape :math:`(*, C, H \times r, W \times r)` to a te... |
| `torch.poisson` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | poisson(input, generator=None) -> Tensor Returns a tensor of the same size as :attr:`input` with each element sampled from a Poisson distribution with rate parameter given by the corresponding elem... |
| `torch.poisson_nll_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.polar` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | polar(abs, angle, *, out=None) -> Tensor Constructs a complex tensor whose elements are Cartesian coordinates corresponding to the polar coordinates with absolute value :attr:`abs` and angle :attr:... |
| `torch.polygamma` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | polygamma(n, input, *, out=None) -> Tensor Alias for :func:`torch.special.polygamma`. |
| `torch.positive` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | positive(input) -> Tensor Returns :attr:`input`. Throws a runtime error if :attr:`input` is a bool tensor. Args: input (Tensor): the input tensor. Example:: >>> t = torch.randn(5) >>> t tensor([ 0.... |
| `torch.pow` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | pow(input, exponent, *, out=None) -> Tensor Takes the power of each element in :attr:`input` with :attr:`exponent` and returns a tensor with the result. :attr:`exponent` can be either a single ``fl... |
| `torch.prod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | prod(input: Tensor, *, dtype: Optional[_dtype]) -> Tensor Returns the product of all elements in the :attr:`input` tensor. Args: input (Tensor): the input tensor. Keyword args: dtype (:class:`torch... |
| `torch.promote_types` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | promote_types(type1, type2) -> dtype Returns the :class:`torch.dtype` with the smallest size and scalar kind that is not smaller nor of lower kind than either `type1` or `type2`. See type promotion... |
| `torch.put` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.q_per_channel_axis` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.q_per_channel_scales` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.q_per_channel_zero_points` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.q_scale` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.q_zero_point` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.qr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | qr(input: Tensor, some: bool = True, *, out: Union[Tensor, Tuple[Tensor, ...], List[Tensor], None]) -> (Tensor, Tensor) Computes the QR decomposition of a matrix or a batch of matrices :attr:`input... |
| `torch.qscheme` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.quantile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | quantile(input, q, dim=None, keepdim=False, *, interpolation='linear', out=None) -> Tensor Computes the q-th quantiles of each row of the :attr:`input` tensor along the dimension :attr:`dim`. To co... |
| `torch.quantize_per_channel` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | quantize_per_channel(input, scales, zero_points, axis, dtype) -> Tensor Converts a float tensor to a per-channel quantized tensor with given scales and zero points. Arguments: input (Tensor): float... |
| `torch.quantized_batch_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | quantized_batch_norm(input, weight=None, bias=None, mean, var, eps, output_scale, output_zero_point) -> Tensor Applies batch normalization on a 4D (NCHW) quantized tensor. .. math:: y = \frac{x - \... |
| `torch.quantized_gru_cell` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.quantized_lstm_cell` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.rad2deg` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | rad2deg(input: Tensor, *, out: Optional[Tensor]) -> Tensor Returns a new tensor with each of the elements of :attr:`input` converted from angles in radians to degrees. Args: input (Tensor): the inp... |
| `torch.rad2deg_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.range` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | range(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Returns a 1-D tensor of size :math:`\left\lfloor \frac{\text{end} - \text{star... |
| `torch.ravel` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ravel(input) -> Tensor Return a contiguous flattened tensor. A copy is made only if needed. Args: input (Tensor): the input tensor. Example:: >>> t = torch.tensor([[[1, 2], ... [3, 4]], ... [[5, 6]... |
| `torch.read_vitals` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | read_vitals() -> str |
| `torch.real` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | real(input) -> Tensor Returns a new tensor containing real values of the :attr:`self` tensor. The returned tensor and :attr:`self` share the same underlying storage. Args: input (Tensor): the input... |
| `torch.reciprocal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | reciprocal(input, *, out=None) -> Tensor Returns a new tensor with the reciprocal of the elements of :attr:`input` .. math:: \text{out}_{i} = \frac{1}{\text{input}_{i}} .. note:: Unlike NumPy's rec... |
| `torch.reciprocal_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.remainder` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | remainder(input, other, *, out=None) -> Tensor Computes `Python's modulus operation <https://docs.python.org/3/reference/expressions.html#binary-arithmetic-operations>`_ entrywise. The result has t... |
| `torch.renorm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | renorm(input, p, dim, maxnorm, *, out=None) -> Tensor Returns a tensor where each sub-tensor of :attr:`input` along dimension :attr:`dim` is normalized such that the `p`-norm of the sub-tensor is l... |
| `torch.repeat_interleave` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | repeat_interleave(input, repeats, dim=None, *, output_size=None) -> Tensor Repeat elements of a tensor. .. warning:: This is different from :meth:`torch.Tensor.repeat` but similar to ``numpy.repeat... |
| `torch.reshape` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | reshape(input, shape) -> Tensor Returns a tensor with the same data and number of elements as :attr:`input`, but with the specified shape. When possible, the returned tensor will be a view of :attr... |
| `torch.resize_as_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.resize_as_sparse_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.resolve_conj` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | resolve_conj(input) -> Tensor Returns a new tensor with materialized conjugation if :attr:`input`'s conjugate bit is set to `True`, else returns :attr:`input`. The output tensor will always have it... |
| `torch.resolve_neg` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | resolve_neg(input) -> Tensor Returns a new tensor with materialized negation if :attr:`input`'s negative bit is set to `True`, else returns :attr:`input`. The output tensor will always have its neg... |
| `torch.result_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | result_type(tensor1, tensor2) -> dtype Returns the :class:`torch.dtype` that would result from performing an arithmetic operation on the provided input tensors. See type promotion :ref:`documentati... |
| `torch.rms_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.roll` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | roll(input, shifts, dims=None) -> Tensor Roll the tensor :attr:`input` along the given dimension(s). Elements that are shifted beyond the last position are re-introduced at the first position. If :... |
| `torch.rot90` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | rot90(input, k=1, dims=(0, 1)) -> Tensor Rotate an n-D tensor by 90 degrees in the plane specified by dims axis. Rotation direction is from the first towards the second axis if k > 0, and from the ... |
| `torch.round` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | round(input, *, decimals=0, out=None) -> Tensor Rounds elements of :attr:`input` to the nearest integer. For integer inputs, follows the array-api convention of returning a copy of the input tensor... |
| `torch.round_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.row_indices_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.row_stack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | row_stack(tensors, *, out=None) -> Tensor Alias of :func:`torch.vstack`. |
| `torch.rsub` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.saddmm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.save` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj, f, pickle_module, ...` | `None` | save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True) Saves an object to a disk file. See also: :ref:`saving-loading-tensors` Args: obj: saved object f: a file-... |
| `torch.scatter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | scatter(input, dim, index, src) -> Tensor Out-of-place version of :meth:`torch.Tensor.scatter_` |
| `torch.scatter_add` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | scatter_add(input, dim, index, src) -> Tensor Out-of-place version of :meth:`torch.Tensor.scatter_add_` |
| `torch.scatter_reduce` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | scatter_reduce(input, dim, index, src, reduce, *, include_self=True) -> Tensor Out-of-place version of :meth:`torch.Tensor.scatter_reduce_` |
| `torch.searchsorted` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | searchsorted(sorted_sequence, values, *, out_int32=False, right=False, side=None, out=None, sorter=None) -> Tensor Find the indices from the *innermost* dimension of :attr:`sorted_sequence` such th... |
| `torch.seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Sets the seed for generating random numbers to a non-deterministic random number on all devices. Returns a 64 bit number used to seed the RNG. |
| `torch.segment_reduce` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.select` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | select(input, dim, index) -> Tensor Slices the :attr:`input` tensor along the selected dimension at the given index. This function returns a view of the original tensor with the given dimension rem... |
| `torch.select_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.select`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.select_scatter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | select_scatter(input, src, dim, index) -> Tensor Embeds the values of the :attr:`src` tensor into :attr:`input` at the given index. This function returns a tensor with fresh storage; it does not cr... |
| `torch.selu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.selu_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | selu_(input) -> Tensor In-place version of :func:`~selu`. |
| `torch.set_anomaly_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.set_autocast_cache_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.set_autocast_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.set_autocast_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.set_autocast_gpu_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.set_autocast_ipu_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.set_autocast_ipu_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.set_autocast_xla_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.set_autocast_xla_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.set_default_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `d` | `None` | Sets the default floating point dtype to :attr:`d`. Supports floating point dtype as inputs. Other dtypes will cause torch to raise an exception. When PyTorch is initialized its default floating po... |
| `torch.set_deterministic_debug_mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `debug_mode` | `None` | Sets the debug mode for deterministic operations. .. note:: This is an alternative interface for :func:`torch.use_deterministic_algorithms`. Refer to that function's documentation for details about... |
| `torch.set_float32_matmul_precision` | ❓ | ❓ | ❓ | ❓ | 🔴 | `precision` | `None` | Sets the internal precision of float32 matrix multiplications. Running float32 matrix multiplications in lower precision may significantly increase performance, and in some programs the loss of pre... |
| `torch.set_grad_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mode` | `None` | Context-manager that sets gradient calculation on or off. ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`. It can be used as a context-manager or as a function.... |
| `torch.set_num_interop_threads` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | set_num_interop_threads(int) Sets the number of threads used for interop parallelism (e.g. in JIT interpreter) on CPU. .. warning:: Can only be called once and before any inter-op parallel work is ... |
| `torch.set_num_threads` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | set_num_threads(int) Sets the number of threads used for intraop parallelism on CPU. .. warning:: To ensure that the correct number of threads is used, set_num_threads must be called before running... |
| `torch.set_printoptions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `precision, threshold, edgeitems, ...` | `Any` | Set options for printing. Items shamelessly taken from NumPy Args: precision: Number of digits of precision for floating point output (default = 4). threshold: Total number of array elements which ... |
| `torch.set_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `new_state` | `None` | Sets the random number generator state. .. note:: This function only works for CPU. For CUDA, please use :func:`torch.manual_seed`, which works for both CPU and CUDA. Args: new_state (torch.ByteTen... |
| `torch.set_vital` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | set_vital(arg0: str, arg1: str, arg2: str) -> bool |
| `torch.set_warn_always` | ❓ | ❓ | ❓ | ❓ | 🔴 | `b` | `None` | When this flag is False (default) then some PyTorch warnings may only appear once per process. This helps avoid excessive warning information. Setting it to True causes these warnings to always app... |
| `torch.sgn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sgn(input, *, out=None) -> Tensor This function is an extension of torch.sign() to complex tensors. It computes a new tensor whose elements have the same angles as the corresponding elements of :at... |
| `torch.sign` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sign(input, *, out=None) -> Tensor Returns a new tensor with the signs of the elements of :attr:`input`. .. math:: \text{out}_{i} = \operatorname{sgn}(\text{input}_{i}) Args: input (Tensor): the in... |
| `torch.signbit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | signbit(input, *, out=None) -> Tensor Tests if each element of :attr:`input` has its sign bit set or not. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output te... |
| `torch.slice_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.slice`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.slice_inverse` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.slice_scatter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | slice_scatter(input, src, dim=0, start=None, end=None, step=1) -> Tensor Embeds the values of the :attr:`src` tensor into :attr:`input` at the given dimension. This function returns a tensor with f... |
| `torch.smm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | smm(input, mat) -> Tensor Performs a matrix multiplication of the sparse matrix :attr:`input` with the dense matrix :attr:`mat`. Args: input (Tensor): a sparse matrix to be matrix multiplied mat (T... |
| `torch.solve` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, A, out` | `tuple[torch.Tensor, torch.Tensor]` |  |
| `torch.sort` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sort(input, dim=-1, descending=False, stable=False, *, out=None) -> (Tensor, LongTensor) Sorts the elements of the :attr:`input` tensor along a given dimension in ascending order by value. If :attr... |
| `torch.split` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, split_size_or_sections, dim` | `tuple[torch.Tensor, ...]` | Splits the tensor into chunks. Each chunk is a view of the original tensor. If :attr:`split_size_or_sections` is an integer type, then :attr:`tensor` will be split into equally sized chunks (if pos... |
| `torch.split_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.split`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.split_with_sizes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.split_with_sizes_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.split_with_sizes`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.spmm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.square` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | square(input: Tensor, *, out: Optional[Tensor]) -> Tensor Returns a new tensor with the square of the elements of :attr:`input`. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, o... |
| `torch.square_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.squeeze` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | squeeze(input: Tensor, dim: Optional[Union[int, List[int]]]) -> Tensor Returns a tensor with all specified dimensions of :attr:`input` of size `1` removed. For example, if `input` is of shape: :mat... |
| `torch.squeeze_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.squeeze`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.sspaddmm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sspaddmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) -> Tensor Matrix multiplies a sparse tensor :attr:`mat1` with a dense tensor :attr:`mat2`, then adds the sparse tensor :attr:`input` to the... |
| `torch.stack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | stack(tensors, dim=0, *, out=None) -> Tensor Concatenates a sequence of tensors along a new dimension. All tensors need to be of the same size. .. seealso:: :func:`torch.cat` concatenates the given... |
| `torch.std` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | std(input, dim=None, *, correction=1, keepdim=False, out=None) -> Tensor Calculates the standard deviation over the dimensions specified by :attr:`dim`. :attr:`dim` can be a single dimension, list ... |
| `torch.std_mean` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | std_mean(input, dim=None, *, correction=1, keepdim=False, out=None) -> (Tensor, Tensor) Calculates the standard deviation and mean over the dimensions specified by :attr:`dim`. :attr:`dim` can be a... |
| `torch.stft` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, n_fft, hop_length, ...` | `<class 'torch.Tensor'>` | Short-time Fourier transform (STFT). .. warning:: From version 1.8.0, :attr:`return_complex` must always be given explicitly for real inputs and `return_complex=False` has been deprecated. Strongly... |
| `torch.sub` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sub(input, other, *, alpha=1, out=None) -> Tensor Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`. .. math:: \text{{out}}_i = \text{{input}}_i - \text{{alpha}} \times \text{{ot... |
| `torch.subtract` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | subtract(input, other, *, alpha=1, out=None) -> Tensor Alias for :func:`torch.sub`. |
| `torch.sum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sum(input, *, dtype=None) -> Tensor Returns the sum of all elements in the :attr:`input` tensor. Args: input (Tensor): the input tensor. Keyword args: dtype (:class:`torch.dtype`, optional): the de... |
| `torch.svd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | svd(input, some=True, compute_uv=True, *, out=None) -> (Tensor, Tensor, Tensor) Computes the singular value decomposition of either a matrix or batch of matrices :attr:`input`. The singular value d... |
| `torch.svd_lowrank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `A, q, niter, ...` | `tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | Return the singular value decomposition ``(U, S, V)`` of a matrix, batches of matrices, or a sparse matrix :math:`A` such that :math:`A \approx U \operatorname{diag}(S) V^{\text{H}}`. In case :math... |
| `torch.swapaxes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | swapaxes(input, axis0, axis1) -> Tensor Alias for :func:`torch.transpose`. This function is equivalent to NumPy's swapaxes function. Examples:: >>> x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]]) >... |
| `torch.swapdims` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | swapdims(input, dim0, dim1) -> Tensor Alias for :func:`torch.transpose`. This function is equivalent to NumPy's swapaxes function. Examples:: >>> x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]]) >>>... |
| `torch.sym_constrain_range` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sym_constrain_range_for_size` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sym_float` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` | SymInt-aware utility for float casting. Args: a (SymInt, SymFloat, or object): Object to cast |
| `torch.sym_fresh_size` | ❓ | ❓ | ❓ | ❓ | 🔴 | `expr` | `Any` |  |
| `torch.sym_int` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` | SymInt-aware utility for int casting. Args: a (SymInt, SymFloat, or object): Object to cast |
| `torch.sym_ite` | ❓ | ❓ | ❓ | ❓ | 🔴 | `b, t, f` | `Any` |  |
| `torch.sym_max` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `Any` | SymInt-aware utility for max which avoids branching on a < b. Unlike builtins.max(), this only works for int/float, and it always promotes to float if any argument is float (unlike builtins.max, wh... |
| `torch.sym_min` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `Any` | SymInt-aware utility for min(). |
| `torch.sym_not` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` | SymInt-aware utility for logical negation. Args: a (SymBool or bool): Object to negate |
| `torch.sym_sum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args` | `Any` | N-ary add which is faster to compute for long lists than iterated binary addition. Only does something special for integers. |
| `torch.symeig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, eigenvectors, upper, ...` | `tuple[torch.Tensor, torch.Tensor]` |  |
| `torch.t` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | t(input) -> Tensor Expects :attr:`input` to be <= 2-D tensor and transposes dimensions 0 and 1. 0-D and 1-D tensors are returned as is. When input is a 2-D tensor this is equivalent to ``transpose(... |
| `torch.t_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.t`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.take` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | take(input, index) -> Tensor Returns a new tensor with the elements of :attr:`input` at the given indices. The input tensor is treated as if it were viewed as a 1-D tensor. The result takes the sam... |
| `torch.take_along_dim` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | take_along_dim(input, indices, dim=None, *, out=None) -> Tensor Selects values from :attr:`input` at the 1-dimensional indices from :attr:`indices` along the given :attr:`dim`. If :attr:`dim` is No... |
| `torch.threshold` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.threshold_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | threshold_(input, threshold, value) -> Tensor In-place version of :func:`~threshold`. |
| `torch.tile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | tile(input, dims) -> Tensor Constructs a tensor by repeating the elements of :attr:`input`. The :attr:`dims` argument specifies the number of repetitions in each dimension. If :attr:`dims` specifie... |
| `torch.to_dlpack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | to_dlpack(tensor) -> PyCapsule Returns an opaque object (a "DLPack capsule") representing the tensor. .. note:: ``to_dlpack`` is a legacy DLPack interface. The capsule it returns cannot be used for... |
| `torch.topk` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | topk(input, k, dim=None, largest=True, sorted=True, *, out=None) -> (Tensor, LongTensor) Returns the :attr:`k` largest elements of the given :attr:`input` tensor along a given dimension. If :attr:`... |
| `torch.trace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | trace(input) -> Tensor Returns the sum of the elements of the diagonal of the input 2-D matrix. Example:: >>> x = torch.arange(1., 10.).view(3, 3) >>> x tensor([[ 1., 2., 3.], [ 4., 5., 6.], [ 7., ... |
| `torch.transpose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | transpose(input, dim0, dim1) -> Tensor Returns a tensor that is a transposed version of :attr:`input`. The given dimensions :attr:`dim0` and :attr:`dim1` are swapped. If :attr:`input` is a strided ... |
| `torch.transpose_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.transpose`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.trapezoid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | trapezoid(y, x=None, *, dx=None, dim=-1) -> Tensor Computes the `trapezoidal rule <https://en.wikipedia.org/wiki/Trapezoidal_rule>`_ along :attr:`dim`. By default the spacing between elements is as... |
| `torch.trapz` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | trapz(y, x, *, dim=-1) -> Tensor Alias for :func:`torch.trapezoid`. |
| `torch.triangular_solve` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | triangular_solve(b, A, upper=True, transpose=False, unitriangular=False, *, out=None) -> (Tensor, Tensor) Solves a system of equations with a square upper or lower triangular invertible matrix :mat... |
| `torch.tril` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | tril(input, diagonal=0, *, out=None) -> Tensor Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices :attr:`input`, the other elements of the result tensor :attr:`out` a... |
| `torch.tril_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | tril_indices(row, col, offset=0, *, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor Returns the indices of the lower triangular part of a :attr:`row`-by- :attr:`col` matrix in a 2-b... |
| `torch.triplet_margin_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.triu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | triu(input, diagonal=0, *, out=None) -> Tensor Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices :attr:`input`, the other elements of the result tensor :attr:`out` are... |
| `torch.triu_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | triu_indices(row, col, offset=0, *, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor Returns the indices of the upper triangular part of a :attr:`row` by :attr:`col` matrix in a 2-by... |
| `torch.true_divide` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | true_divide(dividend, divisor, *, out) -> Tensor Alias for :func:`torch.div` with ``rounding_mode=None``. |
| `torch.trunc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | trunc(input, *, out=None) -> Tensor Returns a new tensor with the truncated integer values of the elements of :attr:`input`. For integer inputs, follows the array-api convention of returning a copy... |
| `torch.trunc_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.typename` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj` | `<class 'str'>` | String representation of the type of an object. This function returns a fully qualified string representation of an object's type. Args: obj (object): The object whose type to represent Returns: st... |
| `torch.unbind` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | unbind(input, dim=0) -> seq Removes a tensor dimension. Returns a tuple of all slices along a given dimension, already without it. Arguments: input (Tensor): the tensor to unbind dim (int): dimensi... |
| `torch.unbind_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.unbind`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.unflatten` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | unflatten(input, dim, sizes) -> Tensor Expands a dimension of the input tensor over multiple dimensions. .. seealso:: :func:`torch.flatten` the inverse of this function. It coalesces several dimens... |
| `torch.unfold_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.unfold`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.unify_type_list` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | unify_type_list(arg0: list[c10::Type]) -> c10::Type |
| `torch.unique` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> tuple[Tensor, Tensor, Tensor] Returns the unique elements of the input tensor. .. note:: This function is differen... |
| `torch.unique_consecutive` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Eliminates all but the first element from every consecutive group of equivalent elements. .. note:: This function is different from :func:`torch.unique` in the sense that this function only elimina... |
| `torch.unravel_index` | ❓ | ❓ | ❓ | ❓ | 🔴 | `indices, shape` | `tuple[torch.Tensor, ...]` | Converts a tensor of flat indices into a tuple of coordinate tensors that index into an arbitrary tensor of the specified shape. Args: indices (Tensor): An integer tensor containing indices into th... |
| `torch.unsafe_chunk` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | unsafe_chunk(input, chunks, dim=0) -> List of Tensors Works like :func:`torch.chunk` but without enforcing the autograd restrictions on inplace modification of the outputs. .. warning:: This functi... |
| `torch.unsafe_split` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | unsafe_split(tensor, split_size_or_sections, dim=0) -> List of Tensors Works like :func:`torch.split` but without enforcing the autograd restrictions on inplace modification of the outputs. .. warn... |
| `torch.unsafe_split_with_sizes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.unsqueeze` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | unsqueeze(input, dim) -> Tensor Returns a new tensor with a dimension of size one inserted at the specified position. The returned tensor shares the same underlying data with this tensor. A :attr:`... |
| `torch.unsqueeze_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.unsqueeze`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.use_deterministic_algorithms` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mode, warn_only` | `None` | Sets whether PyTorch operations must use "deterministic" algorithms. That is, algorithms which, given the same input, and when run on the same software and hardware, always produce the same output.... |
| `torch.values_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.values`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.vander` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | vander(x, N=None, increasing=False) -> Tensor Generates a Vandermonde matrix. The columns of the output matrix are elementwise powers of the input vector :math:`x^{(N-1)}, x^{(N-2)}, ..., x^0`. If ... |
| `torch.var` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | var(input, dim=None, *, correction=1, keepdim=False, out=None) -> Tensor Calculates the variance over the dimensions specified by :attr:`dim`. :attr:`dim` can be a single dimension, list of dimensi... |
| `torch.var_mean` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | var_mean(input, dim=None, *, correction=1, keepdim=False, out=None) -> (Tensor, Tensor) Calculates the variance and mean over the dimensions specified by :attr:`dim`. :attr:`dim` can be a single di... |
| `torch.vdot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | vdot(input, other, *, out=None) -> Tensor Computes the dot product of two 1D vectors along a dimension. In symbols, this function computes .. math:: \sum_{i=1}^n \overline{x_i}y_i. where :math:`\ov... |
| `torch.view_as_complex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | view_as_complex(input) -> Tensor Returns a view of :attr:`input` as a complex tensor. For an input complex tensor of :attr:`size` :math:`m1, m2, \dots, mi, 2`, this function returns a new complex t... |
| `torch.view_as_complex_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.view_as_complex`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.view_as_real` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | view_as_real(input) -> Tensor Returns a view of :attr:`input` as a real tensor. For an input complex tensor of :attr:`size` :math:`m1, m2, \dots, mi`, this function returns a new real tensor of siz... |
| `torch.view_as_real_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.view_as_real`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.view_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.view`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.vitals_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | vitals_enabled() -> bool |
| `torch.vmap` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, in_dims, out_dims, ...` | `typing.Callable` | vmap is the vectorizing map; ``vmap(func)`` returns a new function that maps ``func`` over some dimension of the inputs. Semantically, vmap pushes the map into PyTorch operations called by ``func``... |
| `torch.vsplit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | vsplit(input, indices_or_sections) -> List of Tensors Splits :attr:`input`, a tensor with two or more dimensions, into multiple tensors vertically according to :attr:`indices_or_sections`. Each spl... |
| `torch.vstack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | vstack(tensors, *, out=None) -> Tensor Stack tensors in sequence vertically (row wise). This is equivalent to concatenation along the first axis after all 1-D tensors have been reshaped by :func:`t... |
| `torch.where` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | where(condition, input, other, *, out=None) -> Tensor Return a tensor of elements selected from either :attr:`input` or :attr:`other`, depending on :attr:`condition`. The operation is defined as: .... |
| `torch.while_loop` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cond_fn, body_fn, carried_inputs` | `Any` | Run body_fn(*carried_inputs) while cond_fn(*carried_inputs) returns a True scalar tensor. Returns the output of body_fn or initial carried_inputs. .. warning:: `torch.while_loop` is a prototype fea... |
| `torch.zero_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| | | | | | | | | |
| 🟦 TORCH_EVENTS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.AwaitType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Event` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Event(device, *, enable_timing) -> Event Query and record Stream status to identify or control dependencies across Stream and measure timing. Arguments: device (:class:`torch.device`, optional): th... |
| `torch.wait` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | wait(arg0: torch._C.Future) -> object |
| | | | | | | | | |
| 🟦 TORCH_TENSOR_CREATION | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.BFloat16Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.BoolTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ByteTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.CharTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.DoubleTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.FloatTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.HalfTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.IntTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.LongTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ShortTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.TensorType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.align_tensors` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` |  |
| `torch.arange` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Returns a 1-D tensor of size :math:`\left\lceil \frac{\text{end} - \text{star... |
| `torch.as_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | as_tensor(data: Any, dtype: Optional[dtype] = None, device: Optional[DeviceLikeType]) -> Tensor Converts :attr:`data` into a tensor, sharing data and preserving autograd history if possible. If :at... |
| `torch.broadcast_tensors` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` | broadcast_tensors(*tensors) -> List of Tensors Broadcasts the given tensors according to :ref:`broadcasting-semantics`. Args: *tensors: any number of tensors of the same type .. warning:: More than... |
| `torch.empty` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format) -> Tensor Returns a tensor filled with uniniti... |
| `torch.empty_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor Returns an uninitialized tensor with the same size as :attr:`input`. `... |
| `torch.empty_permuted` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | empty_permuted(size, physical_layout, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) -> Tensor Creates an uninitialized, non-overlapping and dense tensor with the s... |
| `torch.empty_quantized` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.empty_strided` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | empty_strided(size, stride, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) -> Tensor Creates a tensor with the specified :attr:`size` and :attr:`stride` and filled ... |
| `torch.eye` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Returns a 2-D tensor with ones on the diagonal and zeros elsewhere. Args: n (int): the numb... |
| `torch.fake_quantize_per_tensor_affine` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max) -> Tensor Returns a new tensor with the data in :attr:`input` fake quantized using :attr:`scale`, :attr:`zero_point`,... |
| `torch.is_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj` | `typing_extensions.TypeIs[ForwardRef('torch.Tensor')]` | Returns True if `obj` is a PyTorch tensor. Note that this function is simply doing ``isinstance(obj, Tensor)``. Using that ``isinstance`` check is better for typechecking with mypy, and more explic... |
| `torch.linspace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Creates a one-dimensional tensor of size :attr:`steps` whose values are evenly... |
| `torch.ones` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Returns a tensor filled with the scalar value `1`, with the shape defined by the variable argu... |
| `torch.ones_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor Returns a tensor filled with the scalar value `1`, with the same size a... |
| `torch.quantize_per_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | quantize_per_tensor(input, scale, zero_point, dtype) -> Tensor Converts a float tensor to a quantized tensor with given scale and zero point. Arguments: input (Tensor): float tensor or list of tens... |
| `torch.quantize_per_tensor_dynamic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | quantize_per_tensor_dynamic(input, dtype, reduce_range) -> Tensor Converts a float tensor to a quantized tensor with scale and zero_point calculated dynamically based on the input. Arguments: input... |
| `torch.scalar_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.set_default_tensor_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `t` | `None` | .. warning:: This function is deprecated as of PyTorch 2.1, please use :func:`torch.set_default_dtype()` and :func:`torch.set_default_device()` as alternatives. Sets the default ``torch.Tensor`` ty... |
| `torch.sparse_bsc_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sparse_bsc_tensor(ccol_indices, row_indices, values, size=None, *, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None) -> Tensor Constructs a :ref:`sparse tensor ... |
| `torch.sparse_bsr_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sparse_bsr_tensor(crow_indices, col_indices, values, size=None, *, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None) -> Tensor Constructs a :ref:`sparse tensor ... |
| `torch.sparse_compressed_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sparse_compressed_tensor(compressed_indices, plain_indices, values, size=None, *, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None) -> Tensor Const... |
| `torch.sparse_coo_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sparse_coo_tensor(indices, values, size=None, *, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None, is_coalesced=None) -> Tensor Constructs a :ref:`sparse tensor... |
| `torch.sparse_csc_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sparse_csc_tensor(ccol_indices, row_indices, values, size=None, *, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None) -> Tensor Constructs a :ref:`sparse tensor ... |
| `torch.sparse_csr_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sparse_csr_tensor(crow_indices, col_indices, values, size=None, *, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None) -> Tensor Constructs a :ref:`sparse tensor ... |
| `torch.tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor Constructs a tensor with no autograd history (also known as a "leaf tensor", see :doc:`/notes/autograd`) by... |
| `torch.tensor_split` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | tensor_split(input, indices_or_sections, dim=0) -> List of Tensors Splits a tensor into multiple sub-tensors, all of which are views of :attr:`input`, along dimension :attr:`dim` according to the i... |
| `torch.tensordot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b, dims, ...` | `Any` | Returns a contraction of a and b over multiple dimensions. :attr:`tensordot` implements a generalized matrix product. Args: a (Tensor): Left tensor to contract b (Tensor): Right tensor to contract ... |
| `torch.zeros` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Returns a tensor filled with the scalar value `0`, with the shape defined by the variable arg... |
| `torch.zeros_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor Returns a tensor filled with the scalar value `0`, with the same size ... |
| | | | | | | | | |
| 🟦 TORCH_DEVICE | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.DeviceObjType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.get_autocast_cpu_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.get_default_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `torch.device` | Gets the default ``torch.Tensor`` to be allocated on ``device`` |
| `torch.get_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.is_autocast_cpu_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.profiler_allow_cudagraph_cupti_lazy_reinit_cuda12` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.set_autocast_cpu_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.set_autocast_cpu_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.set_default_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Sets the default ``torch.Tensor`` to be allocated on ``device``. This does not affect factory function calls which are called with an explicit ``device`` argument. Factory calls will be performed a... |
| | | | | | | | | |
| 🟦 TORCH_MATH_OPS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.LockingLogger` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.LoggerBase` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.NoopLogger` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.abs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | abs(input: Tensor, *, out: Optional[Tensor]) -> Tensor Computes the absolute value of each element in :attr:`input`. .. math:: \text{out}_{i} = |\text{input}_{i}| Args: input (Tensor): the input te... |
| `torch.abs_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.absolute` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | absolute(input: Tensor, *, out: Optional[Tensor]) -> Tensor Alias for :func:`torch.abs` |
| `torch.acos` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | acos(input: Tensor, *, out: Optional[Tensor]) -> Tensor Computes the inverse cosine of each element in :attr:`input`. .. math:: \text{out}_{i} = \cos^{-1}(\text{input}_{i}) Args: input (Tensor): th... |
| `torch.acos_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.acosh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | acosh(input: Tensor, *, out: Optional[Tensor]) -> Tensor Returns a new tensor with the inverse hyperbolic cosine of the elements of :attr:`input`. .. math:: \text{out}_{i} = \cosh^{-1}(\text{input}... |
| `torch.acosh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.arccos` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | arccos(input: Tensor, *, out: Optional[Tensor]) -> Tensor Alias for :func:`torch.acos`. |
| `torch.arccos_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.arccosh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | arccosh(input: Tensor, *, out: Optional[Tensor]) -> Tensor Alias for :func:`torch.acosh`. |
| `torch.arccosh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.arcsin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | arcsin(input: Tensor, *, out: Optional[Tensor]) -> Tensor Alias for :func:`torch.asin`. |
| `torch.arcsin_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.arcsinh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | arcsinh(input: Tensor, *, out: Optional[Tensor]) -> Tensor Alias for :func:`torch.asinh`. |
| `torch.arcsinh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.arctan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | arctan(input: Tensor, *, out: Optional[Tensor]) -> Tensor Alias for :func:`torch.atan`. |
| `torch.arctan2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | arctan2(input: Tensor, other: Tensor, *, out: Optional[Tensor]) -> Tensor Alias for :func:`torch.atan2`. |
| `torch.arctan_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.arctanh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | arctanh(input: Tensor, *, out: Optional[Tensor]) -> Tensor Alias for :func:`torch.atanh`. |
| `torch.arctanh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.asin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | asin(input: Tensor, *, out: Optional[Tensor]) -> Tensor Returns a new tensor with the arcsine of the elements of :attr:`input`. .. math:: \text{out}_{i} = \sin^{-1}(\text{input}_{i}) Args: input (T... |
| `torch.asin_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.asinh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | asinh(input: Tensor, *, out: Optional[Tensor]) -> Tensor Returns a new tensor with the inverse hyperbolic sine of the elements of :attr:`input`. .. math:: \text{out}_{i} = \sinh^{-1}(\text{input}_{... |
| `torch.asinh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.atan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | atan(input: Tensor, *, out: Optional[Tensor]) -> Tensor Returns a new tensor with the arctangent of the elements of :attr:`input`. .. math:: \text{out}_{i} = \tan^{-1}(\text{input}_{i}) Args: input... |
| `torch.atan2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | atan2(input: Tensor, other: Tensor, *, out: Optional[Tensor]) -> Tensor Element-wise arctangent of :math:`\text{input}_{i} / \text{other}_{i}` with consideration of the quadrant. Returns a new tens... |
| `torch.atan_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.atanh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | atanh(input: Tensor, *, out: Optional[Tensor]) -> Tensor Returns a new tensor with the inverse hyperbolic tangent of the elements of :attr:`input`. Note: The domain of the inverse hyperbolic tangen... |
| `torch.atanh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.binary_cross_entropy_with_logits` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.ceil` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ceil(input, *, out=None) -> Tensor Returns a new tensor with the ceil of the elements of :attr:`input`, the smallest integer greater than or equal to each element. For integer inputs, follows the a... |
| `torch.ceil_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.constant_pad_nd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cos` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cos(input, *, out=None) -> Tensor Returns a new tensor with the cosine of the elements of :attr:`input`. .. math:: \text{out}_{i} = \cos(\text{input}_{i}) Args: input (Tensor): the input tensor. Ke... |
| `torch.cos_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cosh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cosh(input, *, out=None) -> Tensor Returns a new tensor with the hyperbolic cosine of the elements of :attr:`input`. .. math:: \text{out}_{i} = \cosh(\text{input}_{i}) Args: input (Tensor): the inp... |
| `torch.cosh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cosine_embedding_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cosine_similarity` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cosine_similarity(x1, x2, dim=1, eps=1e-8) -> Tensor Returns cosine similarity between ``x1`` and ``x2``, computed along dim. ``x1`` and ``x2`` must be broadcastable to a common shape. ``dim`` refe... |
| `torch.exp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | exp(input, *, out=None) -> Tensor Returns a new tensor with the exponential of the elements of the input tensor :attr:`input`. .. math:: y_{i} = e^{x_{i}} Args: input (Tensor): the input tensor. Ke... |
| `torch.exp2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | exp2(input, *, out=None) -> Tensor Alias for :func:`torch.special.exp2`. |
| `torch.exp2_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.exp_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.expand_copy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs the same operation as :func:`torch.Tensor.expand`, but all output tensors are freshly created instead of aliasing the input. |
| `torch.expm1` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | expm1(input, *, out=None) -> Tensor Alias for :func:`torch.special.expm1`. |
| `torch.expm1_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.floor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | floor(input, *, out=None) -> Tensor Returns a new tensor with the floor of the elements of :attr:`input`, the largest integer less than or equal to each element. For integer inputs, follows the arr... |
| `torch.floor_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.floor_divide` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | floor_divide(input, other, *, out=None) -> Tensor .. note:: Before PyTorch 1.13 :func:`torch.floor_divide` incorrectly performed truncation division. To restore the previous behavior use :func:`tor... |
| `torch.frexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | frexp(input, *, out=None) -> (Tensor mantissa, Tensor exponent) Decomposes :attr:`input` into mantissa and exponent tensors such that :math:`\text{input} = \text{mantissa} \times 2^{\text{exponent}... |
| `torch.instance_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.isin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | isin(elements, test_elements, *, assume_unique=False, invert=False) -> Tensor Tests if each element of :attr:`elements` is in :attr:`test_elements`. Returns a boolean tensor of the same shape as :a... |
| `torch.isinf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | isinf(input) -> Tensor Tests if each element of :attr:`input` is infinite (positive or negative infinity) or not. .. note:: Complex values are infinite when their real or imaginary part is infinite... |
| `torch.isposinf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | isposinf(input, *, out=None) -> Tensor Tests if each element of :attr:`input` is positive infinity or not. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output t... |
| `torch.ldexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ldexp(input, other, *, out=None) -> Tensor Multiplies :attr:`input` by 2 ** :attr:`other`. .. math:: \text{{out}}_i = \text{{input}}_i * 2^\text{{other}}_i Typically this function is used to constr... |
| `torch.ldexp_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.log` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | log(input, *, out=None) -> Tensor Returns a new tensor with the natural logarithm of the elements of :attr:`input`. .. math:: y_{i} = \log_{e} (x_{i}) Args: input (Tensor): the input tensor. Keywor... |
| `torch.log10` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | log10(input: Tensor, *, out: Optional[Tensor]) -> Tensor Returns a new tensor with the logarithm to the base 10 of the elements of :attr:`input`. .. math:: y_{i} = \log_{10} (x_{i}) Args: input (Te... |
| `torch.log10_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.log1p` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | log1p(input, *, out=None) -> Tensor Returns a new tensor with the natural logarithm of (1 + :attr:`input`). .. math:: y_i = \log_{e} (x_i + 1) .. note:: This function is more accurate than :func:`t... |
| `torch.log1p_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.log2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | log2(input: Tensor, *, out: Optional[Tensor]) -> Tensor Returns a new tensor with the logarithm to the base 2 of the elements of :attr:`input`. .. math:: y_{i} = \log_{2} (x_{i}) Args: input (Tenso... |
| `torch.log2_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.log_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.log_softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.logaddexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logaddexp(input, other, *, out=None) -> Tensor Logarithm of the sum of exponentiations of the inputs. Calculates pointwise :math:`\log\left(e^x + e^y\right)`. This function is useful in statistics ... |
| `torch.logaddexp2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logaddexp2(input, other, *, out=None) -> Tensor Logarithm of the sum of exponentiations of the inputs in base-2. Calculates pointwise :math:`\log_2\left(2^x + 2^y\right)`. See :func:`torch.logaddex... |
| `torch.logcumsumexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logcumsumexp(input, dim, *, out=None) -> Tensor Returns the logarithm of the cumulative summation of the exponentiation of elements of :attr:`input` in the dimension :attr:`dim`. For summation inde... |
| `torch.logdet` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logdet(input) -> Tensor Calculates log determinant of a square matrix or batches of square matrices. It returns ``-inf`` if the input has a determinant of zero, and ``NaN`` if it has a negative det... |
| `torch.logical_and` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logical_and(input, other, *, out=None) -> Tensor Computes the element-wise logical AND of the given input tensors. Zeros are treated as ``False`` and nonzeros are treated as ``True``. Args: input (... |
| `torch.logical_not` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logical_not(input, *, out=None) -> Tensor Computes the element-wise logical NOT of the given input tensor. If not specified, the output tensor will have the bool dtype. If the input tensor is not a... |
| `torch.logical_or` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logical_or(input, other, *, out=None) -> Tensor Computes the element-wise logical OR of the given input tensors. Zeros are treated as ``False`` and nonzeros are treated as ``True``. Args: input (Te... |
| `torch.logical_xor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logical_xor(input: Tensor, other: Tensor, *, out: Optional[Tensor]) -> Tensor Computes the element-wise logical XOR of the given input tensors. Zeros are treated as ``False`` and nonzeros are treat... |
| `torch.logit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logit(input, eps=None, *, out=None) -> Tensor Alias for :func:`torch.special.logit`. |
| `torch.logit_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.logspace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Creates a one-dimensional tensor of size :attr:`steps` whose values... |
| `torch.logsumexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logsumexp(input, dim, keepdim=False, *, out=None) Returns the log of summed exponentials of each row of the :attr:`input` tensor in the given dimension :attr:`dim`. The computation is numerically s... |
| `torch.matrix_exp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | matrix_exp(A) -> Tensor Alias for :func:`torch.linalg.matrix_exp`. |
| `torch.pairwise_distance` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False) -> Tensor See :class:`torch.nn.PairwiseDistance` for details |
| `torch.prepare_multiprocessing_environment` | ❓ | ❓ | ❓ | ❓ | 🔴 | `path` | `None` |  |
| `torch.quantized_rnn_tanh_cell` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.rnn_tanh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.rnn_tanh_cell` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.rsqrt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | rsqrt(input, *, out=None) -> Tensor Returns a new tensor with the reciprocal of the square-root of each of the elements of :attr:`input`. .. math:: \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}... |
| `torch.rsqrt_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sin(input, *, out=None) -> Tensor Returns a new tensor with the sine of the elements of :attr:`input`. .. math:: \text{out}_{i} = \sin(\text{input}_{i}) Args: input (Tensor): the input tensor. Keyw... |
| `torch.sin_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sinc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sinc(input, *, out=None) -> Tensor Alias for :func:`torch.special.sinc`. |
| `torch.sinc_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sinh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sinh(input, *, out=None) -> Tensor Returns a new tensor with the hyperbolic sine of the elements of :attr:`input`. .. math:: \text{out}_{i} = \sinh(\text{input}_{i}) Args: input (Tensor): the input... |
| `torch.sinh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.slogdet` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | slogdet(input) -> (Tensor, Tensor) Alias for :func:`torch.linalg.slogdet` |
| `torch.sqrt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sqrt(input, *, out=None) -> Tensor Returns a new tensor with the square-root of the elements of :attr:`input`. .. math:: \text{out}_{i} = \sqrt{\text{input}_{i}} Args: input (Tensor): the input ten... |
| `torch.sqrt_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sym_sqrt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch.tan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | tan(input, *, out=None) -> Tensor Returns a new tensor with the tangent of the elements of :attr:`input`. .. math:: \text{out}_{i} = \tan(\text{input}_{i}) Args: input (Tensor): the input tensor. K... |
| `torch.tan_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.tanh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | tanh(input, *, out=None) -> Tensor Returns a new tensor with the hyperbolic tangent of the elements of :attr:`input`. .. math:: \text{out}_{i} = \tanh(\text{input}_{i}) Args: input (Tensor): the in... |
| `torch.tanh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.xlogy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | xlogy(input, other, *, out=None) -> Tensor Alias for :func:`torch.special.xlogy`. |
| `torch.xlogy_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| | | | | | | | | |
| 🟦 TORCH_MEMORY | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.OutOfMemoryError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Exception raised when device is out of memory |
| `torch.memory_format` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| | | | | | | | | |
| 🟦 TORCH_STREAMS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.Stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Stream(device, *, priority) -> Stream An in-order queue of executing the respective tasks asynchronously in first in first out (FIFO) order. It can control or synchronize the execution of other Str... |
| `torch.StreamObjType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| | | | | | | | | |
| 🟦 TORCH_NN_OPS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.adaptive_avg_pool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | adaptive_avg_pool1d(input, output_size) -> Tensor Applies a 1D adaptive average pooling over an input signal composed of several input planes. See :class:`~torch.nn.AdaptiveAvgPool1d` for details a... |
| `torch.adaptive_max_pool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.avg_pool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor Applies a 1D average pooling over an input signal composed of several input planes. See :cl... |
| `torch.bilinear` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bilinear(input1, input2, weight, bias=None) -> Tensor Applies a bilinear transformation to the incoming data: :math:`y = x_1^T A x_2 + b` Shape: - input1: :math:`(N, *, H_{in1})` where :math:`H_{in... |
| `torch.conv1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor Applies a 1D convolution over an input signal composed of several input planes. This operator supports :ref:`Te... |
| `torch.conv2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor Applies a 2D convolution over an input image composed of several input planes. This operator supports :ref:`Ten... |
| `torch.conv3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor Applies a 3D convolution over an input image composed of several input planes. This operator supports :ref:`Ten... |
| `torch.conv_tbc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Applies a 1-dimensional sequence convolution over an input sequence. Input and output dimensions are (Time, Batch, Channels) - hence TBC. Args: input: input tensor of shape :math:`(\text{sequence l... |
| `torch.conv_transpose1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor Applies a 1D transposed convolution operator over an input signal composed of sever... |
| `torch.conv_transpose2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor Applies a 2D transposed convolution operator over an input image composed of severa... |
| `torch.conv_transpose3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor Applies a 3D transposed convolution operator over an input image composed of severa... |
| `torch.convolution` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cudnn_convolution` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cudnn_convolution_add_relu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cudnn_convolution_relu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cudnn_convolution_transpose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.fbgemm_linear_fp16_weight` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.fbgemm_linear_fp16_weight_fp32_activation` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.fbgemm_linear_int8_weight` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.fbgemm_linear_int8_weight_fp32_activation` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.fbgemm_linear_quantize_weight` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.max_pool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.max_pool1d_with_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.max_pool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.max_pool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.miopen_convolution` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.miopen_convolution_add_relu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.miopen_convolution_relu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.miopen_convolution_transpose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.miopen_depthwise_convolution` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.mkldnn_adaptive_avg_pool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.mkldnn_convolution` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.mkldnn_linear_backward_weights` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.mkldnn_max_pool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.mkldnn_max_pool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.prelu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | prelu(input, weight) -> Tensor Applies element-wise the function :math:`\text{PReLU}(x) = \max(0,x) + \text{weight} * \min(0,x)` where weight is a learnable parameter. .. note:: `weight` is expecte... |
| `torch.quantized_max_pool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | quantized_max_pool1d(input, kernel_size, stride=[], padding=0, dilation=1, ceil_mode=False) -> Tensor Applies a 1D max pooling over an input quantized tensor composed of several input planes. Argum... |
| `torch.quantized_max_pool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | quantized_max_pool2d(input, kernel_size, stride=[], padding=0, dilation=1, ceil_mode=False) -> Tensor Applies a 2D max pooling over an input quantized tensor composed of several input planes. Argum... |
| `torch.quantized_max_pool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.quantized_rnn_relu_cell` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.relu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.relu_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | relu_(input) -> Tensor In-place version of :func:`~relu`. |
| `torch.rnn_relu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.rnn_relu_cell` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.rrelu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.rrelu_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | rrelu_(input, lower=1./8, upper=1./3, training=False) -> Tensor In-place version of :func:`~rrelu`. |
| `torch.sigmoid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sigmoid(input, *, out=None) -> Tensor Alias for :func:`torch.special.expit`. |
| `torch.sigmoid_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | softmax(input, dim, *, dtype=None) -> Tensor Alias for :func:`torch.nn.functional.softmax`. |
| | | | | | | | | |
| 🟦 TORCH_RANDOM_GENERATION | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.bernoulli` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bernoulli(input: Tensor, *, generator: Optional[Generator], out: Optional[Tensor]) -> Tensor Draws binary random numbers (0 or 1) from a Bernoulli distribution. The :attr:`input` tensor should be a... |
| `torch.normal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | normal(mean, std, *, generator=None, out=None) -> Tensor Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given. The :attr:`mean` is... |
| `torch.rand` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | rand(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor Returns a tensor filled with random numbers from a uniform d... |
| `torch.rand_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | rand_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor Returns a tensor with the same size as :attr:`input` that is filled wit... |
| `torch.randint` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | randint(low=0, high, size, \*, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Returns a tensor filled with random integers generated uniform... |
| `torch.randint_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | randint_like(input, low=0, high, \*, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor Returns a tensor with the same shape as Tenso... |
| `torch.randn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | randn(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor Returns a tensor filled with random numbers from a normal d... |
| `torch.randn_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor Returns a tensor with the same size as :attr:`input` that is filled wi... |
| `torch.randperm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | randperm(n, *, generator=None, out=None, dtype=torch.int64,layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor Returns a random permutation of integers from ``0`` to... |
| `torch.set_flush_denormal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | set_flush_denormal(mode) -> bool Disables denormal floating numbers on CPU. Returns ``True`` if your system supports flushing denormal numbers and it successfully configures flush denormal mode. :m... |
| | | | | | | | | |
| 🟦 CORE_TORCH | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch._awaits.Generic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Abstract base class for generic types. A generic type is typically declared by inheriting from this class parameterized with one or more type variables. For example, a generic mapping type might be... |
| `torch._awaits.TypeVar` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, constraints, bound, ...` | `Any` | Type variable. Usage:: T = TypeVar('T') # Can be anything A = TypeVar('A', str, bytes) # Must be str or bytes Type variables exist primarily for the benefit of static type checkers. They serve as t... |
| `torch._decomp.FunctionalTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `elem, mode` | `Any` | Functional tensors represent tensors that will remove mutations from a program. If you perform a mutable operation on a functional tensor, it will re-dispatch to the functional variant of that oper... |
| `torch._decomp.HigherOrderOperator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, cacheable` | `Any` | Base class for OpOverload (which represents C++ ATen operators) and HigherOrderOperator (which represents Python-only operators that are unrepresentable in TorchScript). |
| `torch._decomp.OpOverload` | ❓ | ❓ | ❓ | ❓ | 🔴 | `overloadpacket, op, op_dk, ...` | `Any` | Base class for OpOverload (which represents C++ ATen operators) and HigherOrderOperator (which represents Python-only operators that are unrepresentable in TorchScript). |
| `torch._decomp.OpOverloadPacket` | ❓ | ❓ | ❓ | ❓ | 🔴 | `qualified_op_name, op_name, op, ...` | `Any` |  |
| `torch._decomp.OperatorBase` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Base class for OpOverload (which represents C++ ATen operators) and HigherOrderOperator (which represents Python-only operators that are unrepresentable in TorchScript). |
| `torch._decomp.ParamSpec` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, bound, covariant, ...` | `Any` | Parameter specification. |
| `torch._decomp.Sequence` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | All the operations on a read-only sequence. Concrete subclasses must override __new__ or __init__, __getitem__, and __len__. |
| `torch._decomp.TypeVar` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, constraints, bound, ...` | `Any` | Type variable. Usage:: T = TypeVar('T') # Can be anything A = TypeVar('A', str, bytes) # Must be str or bytes Type variables exist primarily for the benefit of static type checkers. They serve as t... |
| `torch._decomp.chain` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | chain(*iterables) --> chain object Return a chain object whose .__next__() method returns elements from the first iterable until it is exhausted, then elements from the next iterable, until all of ... |
| `torch._decomp.core_aten_decompositions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `CustomDecompTable` |  |
| `torch._decomp.defaultdict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | defaultdict(default_factory=None, /, [...]) --> dict with default factory The default factory is called without arguments to produce a new value when a key is not present, in __getitem__ only. A de... |
| `torch._decomp.get_decompositions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `aten_ops, type` | `dict[torch._ops.OperatorBase, typing.Callable]` | Retrieve a dictionary of decompositions corresponding to the list of operator overloads and overload packets passed as input. Overload packets will include all decomposed overloads in the packet. I... |
| `torch._decomp.lru_cache` | ❓ | ❓ | ❓ | ❓ | 🔴 | `maxsize, typed` | `Any` | Least-recently-used cache decorator. If *maxsize* is set to None, the LRU features are disabled and the cache can grow without bound. If *typed* is True, arguments of different types will be cached... |
| `torch._decomp.partial` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | partial(func, *args, **keywords) - new function with partial application of the given arguments and keywords. |
| `torch._decomp.register_decomposition` | ❓ | ❓ | ❓ | ❓ | 🔴 | `aten_op, registry, type, ...` | `typing.Callable[[typing.Callable[~_P, ~_T]], typing.Callable[~_P, ~_T]]` | A decorator to register a function as a decomposition to the Python decomposition table. Use it like this:: @register_decomposition(torch.ops.aten.clamp_min) def clamp_min(x): return torch.clamp(se... |
| `torch._decomp.remove_decompositions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `decompositions, aten_ops` | `None` | Given a dictionary of decompositions obtained from get_decompositions(), removes operators associated with a list of operator overloads and overload packets passed as input. If the decomposition di... |
| `torch._decomp.wraps` | ❓ | ❓ | ❓ | ❓ | 🔴 | `wrapped, assigned, updated` | `Any` | Decorator factory to apply update_wrapper() to a wrapper function Returns a decorator that invokes update_wrapper() with the decorated function as the wrapper argument and the arguments to wraps() ... |
| `torch._dynamo.GenerationTracker` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._dynamo.OptimizedModule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod, dynamo_ctx` | `None` | Wraps the original nn.Module object and later patches its forward method to optimized self.forward method. |
| `torch._dynamo.TensorifyState` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
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
| `torch._dynamo.reset_code` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._dynamo.reset_code_caches` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Clears in-memory code cache, which is what stores compiled products. This resets less state than :func:`reset` and is mostly only used for testing purposes. |
| `torch._dynamo.reset_code_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` |  |
| `torch._dynamo.reset_frame_count` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` |  |
| `torch._dynamo.run` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` | Don't do any dynamic compiles, just use prior optimizations |
| `torch._dynamo.set_stance` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stance, skip_guard_eval_unsafe, force_backend` | `None` | Decorator, context manager, function to set the current stance of the compiler. Stances documented in corresponding function in torch/compiler/__init__.py |
| `torch._dynamo.substitute_in_graph` | ❓ | ❓ | ❓ | ❓ | 🔴 | `original_fn, can_constant_fold_through, skip_signature_check, ...` | `typing.Callable[[typing.Callable[~_P, ~_R]], typing.Callable[~_P, ~_R]]` | Register a polyfill handler for a function, usually a C function from the C extension, to be used in place of the original function when inlining the original function in the graph. .. note:: The p... |
| `torch._export.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch._export.ConstantArgument` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, value` | `None` | ConstantArgument(name: str, value: Union[int, float, bool, str, NoneType]) |
| `torch._export.ExportDynamoConfig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `allow_rnn` | `None` | Manage Export-specific configurations of Dynamo. |
| `torch._export.ExportGraphSignature` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input_specs, output_specs` | `None` | :class:`ExportGraphSignature` models the input/output signature of Export Graph, which is a fx.Graph with stronger invariants gurantees. Export Graph is functional and does not access "states" like... |
| `torch._export.InputKind` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ... RED = 1 ... BLUE = 2 ... GREEN = 3 Access them by: - attribute access:: >>> Color.RED <Color.RED: 1> - value... |
| `torch._export.InputSpec` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kind, arg, target, ...` | `None` | InputSpec(kind: torch.export.graph_signature.InputKind, arg: Union[torch.export.graph_signature.TensorArgument, torch.export.graph_signature.SymIntArgument, torch.export.graph_signature.SymFloatArg... |
| `torch._export.OrderedDict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Dictionary that remembers insertion order |
| `torch._export.OutputKind` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ... RED = 1 ... BLUE = 2 ... GREEN = 3 Access them by: - attribute access:: >>> Color.RED <Color.RED: 1> - value... |
| `torch._export.OutputSpec` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kind, arg, target` | `None` | OutputSpec(kind: torch.export.graph_signature.OutputKind, arg: Union[torch.export.graph_signature.TensorArgument, torch.export.graph_signature.SymIntArgument, torch.export.graph_signature.SymFloatA... |
| `torch._export.SymBoolArgument` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name` | `None` | SymBoolArgument(name: str) |
| `torch._export.SymFloatArgument` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name` | `None` | SymFloatArgument(name: str) |
| `torch._export.SymIntArgument` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name` | `None` | SymIntArgument(name: str) |
| `torch._export.TensorArgument` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name` | `None` | TensorArgument(name: str) |
| `torch._export.aot_compile` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f, args, kwargs, ...` | `typing.Union[list[str], str]` | Note: this function is not stable yet Traces either an nn.Module's forward function or just a callable with PyTorch operations inside, generates executable cpp code from the program, and returns th... |
| `torch._export.aot_load` | ❓ | ❓ | ❓ | ❓ | 🔴 | `so_path, device` | `typing.Callable` | Loads a shared library generated by aot_compile and returns a callable Args: so_path: Path to the shared library Returns: A callable |
| `torch._export.compatibility` | ❓ | ❓ | ❓ | ❓ | 🔴 | `is_backward_compatible` | `typing.Callable[[~_T], ~_T]` |  |
| `torch._export.compile_context` | ❓ | ❓ | ❓ | ❓ | 🔴 | `context` | `Any` |  |
| `torch._export.contextmanager` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` | @contextmanager decorator. Typical usage: @contextmanager def some_generator(<arguments>): <setup> try: yield <value> finally: <cleanup> This makes this: with some_generator(<arguments>) as <variab... |
| `torch._export.enable_python_dispatcher` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._export.log_export_usage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kwargs` | `Any` |  |
| `torch._export.lru_cache` | ❓ | ❓ | ❓ | ❓ | 🔴 | `maxsize, typed` | `Any` | Least-recently-used cache decorator. If *maxsize* is set to None, the LRU features are disabled and the cache can grow without bound. If *typed* is True, arguments of different types will be cached... |
| `torch._export.make_fx` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f, decomposition_table, tracing_mode, ...` | `Callable[..., GraphModule]` | Given a function f, return a new function which when executed with valid arguments to f, returns an FX GraphModule representing the set of operations that were executed during the course of execution. |
| `torch._export.patch` | ❓ | ❓ | ❓ | ❓ | 🔴 | `target, new, spec, ...` | `Any` | `patch` acts as a function decorator, class decorator or a context manager. Inside the body of the function or with statement, the `target` is patched with a `new` object. When the function/with st... |
| `torch._export.reorder_kwargs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `user_kwargs, spec` | `dict[str, typing.Any]` | Reorder user-provided kwargs to match the order in `spec`. `spec` is expected to be the in_spec of an exported program, i.e. the spec that results from flattening `(args, kwargs)`. We need this to ... |
| `torch._higher_order_ops.BaseHOP` | ❓ | ❓ | ❓ | ❓ | 🔴 | `hop_name` | `None` | This is the "Base" HOP implementation for a HOP that looks like: call_subgraph_hop(subgraph, *operands, **kwargs) That is: 1) the HOP stays alive until Inductor 2) the HOP's semantics are subgraph(... |
| `torch._higher_order_ops.InvokeQuant` | ❓ | ❓ | ❓ | ❓ | 🔴 | `codegen_low_precision` | `None` | Invoke a quantization function that will be preserved as a single operator. Preservation as a single operator aids in pattern matching and custom lowerings. The operation appears as: torch.ops.high... |
| `torch._higher_order_ops.associative_scan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `combine_fn, xs, dim, ...` | `<class 'torch.Tensor'>` | Performs an inclusive scan with an associative combine function. .. warning:: `torch.associative_scan` is a prototype feature in PyTorch. It currently does not support autograd and you may run into... |
| `torch._higher_order_ops.cond` | ❓ | ❓ | ❓ | ❓ | 🔴 | `pred, true_fn, false_fn, ...` | `typing.Any` | Conditionally applies `true_fn` or `false_fn`. .. warning:: `torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types and doesn't support training currently.... |
| `torch._higher_order_ops.foreach_map` | ❓ | ❓ | ❓ | ❓ | 🔴 | `op, operands, kwargs` | `Any` |  |
| `torch._higher_order_ops.scan` | ❓ | ❓ | ❓ | ❓ | 🔴 | `combine_fn, init, xs, ...` | `tuple[typing.Any, typing.Any]` | Performs an inclusive scan with a combine function. .. warning:: `torch.scan` is a prototype feature in PyTorch. It currently does not support autograd and you may run into miscompiles. Read more a... |
| `torch._higher_order_ops.strict_mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `callable, operands` | `Any` |  |
| `torch._higher_order_ops.while_loop` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cond_fn, body_fn, carried_inputs` | `Any` | Run body_fn(*carried_inputs) while cond_fn(*carried_inputs) returns a True scalar tensor. Returns the output of body_fn or initial carried_inputs. .. warning:: `torch.while_loop` is a prototype fea... |
| `torch._inductor.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch._inductor.CompiledArtifact` | ❓ | ❓ | ❓ | ❓ | 🔴 | `compiled_fn, artifacts` | `Any` | CompiledArtifact class represents the precompiled inductor artifact that can be invoked in order to avoid repeated compilation. CompiledArtifact can be obtained by calling standalone_compile(gm, ex... |
| `torch._inductor.IO` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Generic base class for TextIO and BinaryIO. This is an abstract, generic version of the return of open(). NOTE: This does not distinguish between the different possible classes (text vs. binary, re... |
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
| `torch._logging.LazyString` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, args, kwargs` | `Any` |  |
| `torch._logging.dtrace_structured` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, metadata_fn, payload_fn, ...` | `Any` | For logging more detailed information used for debugging. This may result in the program becoming slow. |
| `torch._logging.getArtifactLogger` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module_qname, artifact_name` | `Any` |  |
| `torch._logging.get_structured_logging_overhead` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `typing.Optional[float]` |  |
| `torch._logging.set_logs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `all, dynamo, aot, ...` | `Any` | Sets the log level for individual components and toggles individual log artifact types. .. warning:: This feature is a prototype and may have compatibility breaking changes in the future. .. note::... |
| `torch._logging.trace_structured` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, metadata_fn, payload_fn, ...` | `None` | metadata is an arbitrary JSON compatible struct, but it's expected to not be too long (e.g., less than 1MB) payload is an arbitrary string, which can be arbitrarily long (but expected to have newli... |
| `torch._numpy.AxisError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Inappropriate argument value (of correct type). |
| `torch._numpy.DType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arg` | `Any` |  |
| `torch._numpy.UFuncTypeError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Inappropriate argument type. |
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
| `torch._numpy.bool_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.broadcast_arrays` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, subok` | `Any` |  |
| `torch._numpy.broadcast_shapes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shapes` | `Any` | broadcast_shapes(*shapes) -> Size Similar to :func:`broadcast_tensors` but for shapes. This is equivalent to ``torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape`` but avoids the need crea... |
| `torch._numpy.broadcast_to` | ❓ | ❓ | ❓ | ❓ | 🔴 | `array, shape, subok` | `Any` |  |
| `torch._numpy.byte` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.can_cast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `from_, to, casting` | `Any` |  |
| `torch._numpy.cbrt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.cdouble` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.ceil` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.cfloat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.choose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, choices, out, ...` | `Any` |  |
| `torch._numpy.clip` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, min, max, ...` | `Any` |  |
| `torch._numpy.column_stack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tup, dtype, casting` | `Any` |  |
| `torch._numpy.common_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors` | `Any` |  |
| `torch._numpy.complex128` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.complex64` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.complex_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.complexfloating` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
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
| `torch._numpy.csingle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
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
| `torch._numpy.double` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
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
| `torch._numpy.float16` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.float32` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.float64` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.float_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.float_power` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.floating` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.floor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.floor_divide` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.fmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.fmin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.fmod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.from_dlpack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `Any` |  |
| `torch._numpy.full` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, fill_value, dtype, ...` | `Any` |  |
| `torch._numpy.full_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, fill_value, dtype, ...` | `Any` |  |
| `torch._numpy.gcd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.generic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.geomspace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `start, stop, num, ...` | `Any` |  |
| `torch._numpy.gradient` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f, varargs, axis, ...` | `Any` |  |
| `torch._numpy.greater` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.greater_equal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.half` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
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
| `torch._numpy.inexact` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.inner` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `Any` |  |
| `torch._numpy.int16` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.int32` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.int64` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.int8` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.int_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.intc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.integer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.intp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
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
| `torch._numpy.longlong` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
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
| `torch._numpy.ndarray` | ❓ | ❓ | ❓ | ❓ | 🔴 | `t` | `Any` |  |
| `torch._numpy.ndim` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch._numpy.negative` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.nextafter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.nonzero` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` |  |
| `torch._numpy.not_equal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x1, x2, out, ...` | `Any` |  |
| `torch._numpy.number` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
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
| `torch._numpy.short` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.sign` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.signbit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.signedinteger` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.sin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, out, where, ...` | `Any` |  |
| `torch._numpy.sinc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `Any` |  |
| `torch._numpy.single` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.singlecomplex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
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
| `torch._numpy.ubyte` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.uint16` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.uint32` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.uint64` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.uint8` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.ulonglong` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.unique` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ar, return_index, return_inverse, ...` | `Any` |  |
| `torch._numpy.unsignedinteger` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` |  |
| `torch._numpy.vander` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x, N, increasing` | `Any` |  |
| `torch._numpy.var` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, axis, dtype, ...` | `Any` |  |
| `torch._numpy.vdot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, b` | `Any` |  |
| `torch._numpy.vsplit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ary, indices_or_sections` | `Any` |  |
| `torch._numpy.vstack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tup, dtype, casting` | `Any` |  |
| `torch._numpy.where` | ❓ | ❓ | ❓ | ❓ | 🔴 | `condition, x, y` | `Any` |  |
| `torch._numpy.zeros` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, dtype, order, ...` | `Any` |  |
| `torch._numpy.zeros_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dtype, order, ...` | `Any` |  |
| `torch._prims.Dim` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | int([x]) -> integer int(x, base=10) -> integer Convert a number or string to an integer, or return 0 if no arguments are given. If x is a number, return x.__int__(). For floating point numbers, thi... |
| `torch._prims.ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ... RED = 1 ... BLUE = 2 ... GREEN = 3 Access them by: - attribute access:: >>> Color.RED <Color.RED: 1> - value... |
| `torch._prims.Enum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ... RED = 1 ... BLUE = 2 ... GREEN = 3 Access them by: - attribute access:: >>> Color.RED <Color.RED: 1> - value... |
| `torch._prims.FakeTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fake_mode, elem, device, ...` | `Self` | Meta tensors give you the ability to run PyTorch code without having to actually do computation through tensors allocated on a `meta` device. Because the device is `meta`, meta tensors do not model... |
| `torch._prims.FakeTensorMode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `allow_fallback_kernels, allow_non_fake_inputs, shape_env, ...` | `None` | A ``TorchDispatchMode`` allows you to override the meaning of all ``__torch_dispatch__`` overrideable functions within a dynamic scope, without having to actually create a tensor subclass or manual... |
| `torch._prims.RETURN_TYPE` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ... RED = 1 ... BLUE = 2 ... GREEN = 3 Access them by: - attribute access:: >>> Color.RED <Color.RED: 1> - value... |
| `torch._prims.Sequence` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | All the operations on a read-only sequence. Concrete subclasses must override __new__ or __init__, __getitem__, and __len__. |
| `torch._prims.Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._prims.TensorLike` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._prims.TensorLikeType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._prims.TensorMeta` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensorlike, shape, strides, ...` | `Any` |  |
| `torch._prims.backwards_not_supported` | ❓ | ❓ | ❓ | ❓ | 🔴 | `prim` | `Any` |  |
| `torch._prims.expand_dims` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, dimensions, ndim` | `<class 'torch.Tensor'>` | Creates a view of a with a.ndim + len(dimensions) dimensions, with new dimensions of length one at the dimensions specified by dimensions. |
| `torch._prims.handle_torch_function` | ❓ | ❓ | ❓ | ❓ | 🔴 | `public_api, relevant_args, args, ...` | `typing.Any` | Implement a function with checks for ``__torch_function__`` overrides. See torch::autograd::handle_torch_function for the equivalent of this function in the C++ implementation. Arguments --------- ... |
| `torch._prims.has_torch_function` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Check for __torch_function__ implementations in the elements of an iterable or if a __torch_function__ mode is enabled. Considers exact ``Tensor`` s and ``Parameter`` s non-dispatchable. Use this t... |
| `torch._prims.is_functional_schema` | ❓ | ❓ | ❓ | ❓ | 🔴 | `schema` | `<class 'bool'>` | Check if the schema is functional. An operator is functional if: - it does not mutate any of its inputs - it does not return a view on any of its inputs - it has at least one return |
| `torch._prims.new_token_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'torch.Tensor'>` |  |
| `torch._prims.partial` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | partial(func, *args, **keywords) - new function with partial application of the given arguments and keywords. |
| `torch._prims.reduce` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | reduce(function, iterable[, initial]) -> value Apply a function of two arguments cumulatively to the items of a sequence or iterable, from left to right, so as to reduce the iterable to a single va... |
| `torch._prims.register_debug_prims` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._prims.register_rng_prims` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._prims.shift_right_logical` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` |  |
| `torch._prims.sym_float` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `Any` | SymInt-aware utility for float casting. Args: a (SymInt, SymFloat, or object): Object to cast |
| `torch._prims.torch_var` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, correction, ...` | `Any` |  |
| `torch._prims.tree_flatten` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tree, is_leaf` | `tuple[list[typing.Any], torch.utils._pytree.TreeSpec]` | Flattens a pytree into a list of values and a TreeSpec that can be used to reconstruct the pytree. |
| `torch._prims.tree_map` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, tree, rests, ...` | `typing.Any` | Map a multi-input function over pytree args to produce a new pytree. See also :func:`tree_map_`. >>> tree_map(lambda x: x + 1, {'x': 7, 'y': (42, 64)}) {'x': 8, 'y': (43, 65)} >>> tree_map(lambda x... |
| `torch._prims.tree_unflatten` | ❓ | ❓ | ❓ | ❓ | 🔴 | `leaves, treespec` | `typing.Any` | Given a list of values and a TreeSpec, builds a pytree. This is the inverse operation of `tree_flatten`. |
| `torch._prims.type_to_dtype` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typ` | `torch.dtype` | Computes the corresponding dtype for a Number type. |
| `torch._prims_common.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch._prims_common.CUDARngStateHelper` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._prims_common.Dim` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | int([x]) -> integer int(x, base=10) -> integer Convert a number or string to an integer, or return 0 if no arguments are given. If x is a number, return x.__int__(). For floating point numbers, thi... |
| `torch._prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ... RED = 1 ... BLUE = 2 ... GREEN = 3 Access them by: - attribute access:: >>> Color.RED <Color.RED: 1> - value... |
| `torch._prims_common.Enum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ... RED = 1 ... BLUE = 2 ... GREEN = 3 Access them by: - attribute access:: >>> Color.RED <Color.RED: 1> - value... |
| `torch._prims_common.FloatWithoutSymFloat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `Any` | Convert a string or number to a floating point number, if possible. |
| `torch._prims_common.IntWithoutSymInt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | int([x]) -> integer int(x, base=10) -> integer Convert a number or string to an integer, or return 0 if no arguments are given. If x is a number, return x.__int__(). For floating point numbers, thi... |
| `torch._prims_common.NamedTuple` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typename, fields, kwargs` | `Any` | Typed version of namedtuple. Usage:: class Employee(NamedTuple): name: str id: int This is equivalent to:: Employee = collections.namedtuple('Employee', ['name', 'id']) The resulting class has an e... |
| `torch._prims_common.REDUCTION_OUTPUT_TYPE_KIND` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ... RED = 1 ... BLUE = 2 ... GREEN = 3 Access them by: - attribute access:: >>> Color.RED <Color.RED: 1> - value... |
| `torch._prims_common.RETURN_TYPE` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ... RED = 1 ... BLUE = 2 ... GREEN = 3 Access them by: - attribute access:: >>> Color.RED <Color.RED: 1> - value... |
| `torch._prims_common.Sequence` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | All the operations on a read-only sequence. Concrete subclasses must override __new__ or __init__, __getitem__, and __len__. |
| `torch._prims_common.Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._prims_common.TensorLike` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._prims_common.TensorLikeType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._prims_common.TypeVar` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, constraints, bound, ...` | `Any` | Type variable. Usage:: T = TypeVar('T') # Can be anything A = TypeVar('A', str, bytes) # Must be str or bytes Type variables exist primarily for the benefit of static type checkers. They serve as t... |
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
| `torch._prims_common.deprecated` | ❓ | ❓ | ❓ | ❓ | 🔴 | `message, category, stacklevel` | `None` | Indicate that a class, function or overload is deprecated. When this decorator is applied to an object, the type checker will generate a diagnostic on usage of the deprecated object. Usage: @deprec... |
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
| `torch._prims_common.nullcontext` | ❓ | ❓ | ❓ | ❓ | 🔴 | `enter_result` | `Any` | Context manager that does no additional processing. Used as a stand-in for a normal context manager, when a particular block of code is only sometimes used with a normal context manager: cm = optio... |
| `torch._prims_common.number_type` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `type` |  |
| `torch._prims_common.overload` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` | Decorator for overloaded functions/methods. In a stub file, place two or more stub definitions for the same function in a row, each decorated with @overload. For example:: @overload def utf8(value:... |
| `torch._prims_common.prod` | ❓ | ❓ | ❓ | ❓ | 🔴 | `xs` | `NumberType` | Product of elements in input sequence. Returns 1 for empty sequence |
| `torch._prims_common.reduce` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | reduce(function, iterable[, initial]) -> value Apply a function of two arguments cumulatively to the items of a sequence or iterable, from left to right, so as to reduce the iterable to a single va... |
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
| `torch._refs.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch._refs.Dim` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | int([x]) -> integer int(x, base=10) -> integer Convert a number or string to an integer, or return 0 if no arguments are given. If x is a number, return x.__int__(). For floating point numbers, thi... |
| `torch._refs.DispatchKey` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: Undefined CompositeExplicitAutogradNonFunctional CompositeExplicitAutograd CompositeImplicitAutogradNestedTensor CompositeImplicitAutograd AutogradNestedTensor AutogradOther Autograd Conju... |
| `torch._refs.ELEMENTWISE_TYPE_PROMOTION_KIND` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ... RED = 1 ... BLUE = 2 ... GREEN = 3 Access them by: - attribute access:: >>> Color.RED <Color.RED: 1> - value... |
| `torch._refs.Enum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ... RED = 1 ... BLUE = 2 ... GREEN = 3 Access them by: - attribute access:: >>> Color.RED <Color.RED: 1> - value... |
| `torch._refs.FloatWithoutSymFloat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `x` | `Any` | Convert a string or number to a floating point number, if possible. |
| `torch._refs.Iterable` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._refs.REDUCTION_OUTPUT_TYPE_KIND` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ... RED = 1 ... BLUE = 2 ... GREEN = 3 Access them by: - attribute access:: >>> Color.RED <Color.RED: 1> - value... |
| `torch._refs.Sequence` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | All the operations on a read-only sequence. Concrete subclasses must override __new__ or __init__, __getitem__, and __len__. |
| `torch._refs.T` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a` | `<class 'torch.Tensor'>` |  |
| `torch._refs.Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._refs.TensorLike` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch._refs.TensorLikeType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
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
| `torch._refs.elementwise_type_promotion_wrapper` | ❓ | ❓ | ❓ | ❓ | 🔴 | `type_promotion_kind, type_promoting_args` | `Any` | Adds elementwise type promotion to a Python reference implementation. Takes two kwargs, type_promoting_args and type_promotion_kind. type_promoting_args must be a string Sequence specifiying the ar... |
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
| `torch._refs.partial` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | partial(func, *args, **keywords) - new function with partial application of the given arguments and keywords. |
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
| `torch._refs.reduce` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | reduce(function, iterable[, initial]) -> value Apply a function of two arguments cumulatively to the items of a sequence or iterable, from left to right, so as to reduce the iterable to a single va... |
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
| `torch._subclasses.CrossRefFakeMode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ignore_op_fn, check_strides, check_aliasing, ...` | `Any` | A ``TorchDispatchMode`` allows you to override the meaning of all ``__torch_dispatch__`` overrideable functions within a dynamic scope, without having to actually create a tensor subclass or manual... |
| `torch._subclasses.DynamicOutputShapeException` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `None` | DynamicOutputShapeException(func: 'OpOverload') |
| `torch._subclasses.FakeTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fake_mode, elem, device, ...` | `Self` | Meta tensors give you the ability to run PyTorch code without having to actually do computation through tensors allocated on a `meta` device. Because the device is `meta`, meta tensors do not model... |
| `torch._subclasses.FakeTensorMode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `allow_fallback_kernels, allow_non_fake_inputs, shape_env, ...` | `None` | A ``TorchDispatchMode`` allows you to override the meaning of all ``__torch_dispatch__`` overrideable functions within a dynamic scope, without having to actually create a tensor subclass or manual... |
| `torch._subclasses.UnsupportedFakeTensorException` | ❓ | ❓ | ❓ | ❓ | 🔴 | `reason` | `None` | UnsupportedFakeTensorException(reason: 'str') |
| | | | | | | | | |
| 🟦 ACCELERATOR_SUPPORT | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.accelerator.current_accelerator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `check_available` | `typing.Optional[torch.device]` | Return the device of the accelerator available at compilation time. If no accelerator were available at compilation time, returns None. See :ref:`accelerator<accelerators>` for details. Args: check... |
| `torch.accelerator.current_device_idx` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the index of a currently selected device for the current :ref:`accelerator<accelerators>`. Returns: int: the index of a currently selected device. |
| `torch.accelerator.current_device_index` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the index of a currently selected device for the current :ref:`accelerator<accelerators>`. Returns: int: the index of a currently selected device. |
| `torch.accelerator.current_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.Stream'>` | Return the currently selected stream for a given device. Args: device (:class:`torch.device`, str, int, optional): a given device that must match the current :ref:`accelerator<accelerators>` device... |
| `torch.accelerator.deprecated` | ❓ | ❓ | ❓ | ❓ | 🔴 | `message, category, stacklevel` | `None` | Indicate that a class, function or overload is deprecated. When this decorator is applied to an object, the type checker will generate a diagnostic on usage of the deprecated object. Usage: @deprec... |
| `torch.accelerator.device_count` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the number of current :ref:`accelerator<accelerators>` available. Returns: int: the number of the current :ref:`accelerator<accelerators>` available. If there is no available accelerators, r... |
| `torch.accelerator.is_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Check if the current accelerator is available at runtime: it was build, all the required drivers are available and at least one device is visible. See :ref:`accelerator<accelerators>` for details. ... |
| `torch.accelerator.set_device_idx` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Set the current device index to a given device. Args: device (:class:`torch.device`, str, int): a given device that must match the current :ref:`accelerator<accelerators>` device type. .. note:: Th... |
| `torch.accelerator.set_device_index` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Set the current device index to a given device. Args: device (:class:`torch.device`, str, int): a given device that must match the current :ref:`accelerator<accelerators>` device type. .. note:: Th... |
| `torch.accelerator.set_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `None` | Set the current stream to a given stream. Args: stream (torch.Stream): a given stream that must match the current :ref:`accelerator<accelerators>` device type. .. note:: This function will set the ... |
| `torch.accelerator.synchronize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Wait for all kernels in all streams on the given device to complete. Args: device (:class:`torch.device`, str, int, optional): device for which to synchronize. It must match the current :ref:`accel... |
| | | | | | | | | |
| 🟦 AUTOMATIC_MIXED_PRECISION | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.amp.GradScaler` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device, init_scale, growth_factor, ...` | `None` | An instance ``scaler`` of :class:`GradScaler`. Helps perform the steps of gradient scaling conveniently. * ``scaler.scale(loss)`` multiplies a given loss by ``scaler``'s current scale factor. * ``s... |
| `torch.amp.autocast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device_type, dtype, enabled, ...` | `Any` | Instances of :class:`autocast` serve as context managers or decorators that allow regions of your script to run in mixed precision. In these regions, ops run in an op-specific dtype chosen by autoc... |
| `torch.amp.custom_bwd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `bwd, device_type` | `Any` | Create a helper decorator for backward methods of custom autograd functions. Autograd functions are subclasses of :class:`torch.autograd.Function`. Ensures that ``backward`` executes with the same ... |
| `torch.amp.custom_fwd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fwd, device_type, cast_inputs` | `Any` | Create a helper decorator for ``forward`` methods of custom autograd functions. Autograd functions are subclasses of :class:`torch.autograd.Function`. See the :ref:`example page<amp-custom-examples... |
| `torch.amp.is_autocast_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device_type` | `<class 'bool'>` | Return a bool indicating if autocast is available on :attr:`device_type`. Args: device_type(str): Device type to use. Possible values are: 'cuda', 'cpu', 'mtia', 'maia', 'xpu', and so on. The type ... |
| | | | | | | | | |
| 🟦 AUTOGRAD | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.autograd.DeviceType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: CPU CUDA MKLDNN OPENGL OPENCL IDEEP HIP FPGA MAIA XLA Vulkan Metal XPU MPS MTIA Meta HPU VE Lazy IPU PrivateUse1 |
| `torch.autograd.Function` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Base class to create custom `autograd.Function`. To create a custom `autograd.Function`, subclass this class and implement the :meth:`forward` and :meth:`backward` static methods. Then, to use your... |
| `torch.autograd.NestedIOFunction` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | This class is here only for backward compatibility reasons. Use :class:`Function` instead of this for any new use case. |
| `torch.autograd.ProfilerActivity` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: CPU XPU MTIA CUDA HPU PrivateUse1 |
| `torch.autograd.ProfilerConfig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.autograd.ProfilerEvent` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.autograd.ProfilerState` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: Disabled CPU CUDA NVTX ITT PRIVATEUSE1 KINETO KINETO_GPU_FALLBACK KINETO_PRIVATEUSE1_FALLBACK |
| `torch.autograd.SavedTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.autograd.Sequence` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | All the operations on a read-only sequence. Concrete subclasses must override __new__ or __init__, __getitem__, and __len__. |
| `torch.autograd.Variable` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.autograd.backward` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensors, grad_tensors, retain_graph, ...` | `None` | Compute the sum of gradients of given tensors with respect to graph leaves. The graph is differentiated using the chain rule. If any of ``tensors`` are non-scalar (i.e. their data has more than one... |
| `torch.autograd.cast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typ, val` | `Any` | Cast a value to a type. This returns the value unchanged. To the type checker this signals that the return value has the designated type, but at runtime we intentionally don't check anything (we wa... |
| `torch.autograd.detect_anomaly` | ❓ | ❓ | ❓ | ❓ | 🔴 | `check_nan` | `None` | Context-manager that enable anomaly detection for the autograd engine. This does two things: - Running the forward pass with detection enabled will allow the backward pass to print the traceback of... |
| `torch.autograd.enable_grad` | ❓ | ❓ | ❓ | ❓ | 🔴 | `orig_func` | `Any` | Context-manager that enables gradient calculation. Enables gradient calculation, if it has been disabled via :class:`~no_grad` or :class:`~set_grad_enabled`. This context manager is thread local; i... |
| `torch.autograd.grad` | ❓ | ❓ | ❓ | ❓ | 🔴 | `outputs, inputs, grad_outputs, ...` | `tuple[torch.Tensor, ...]` | Compute and return the sum of gradients of outputs with respect to the inputs. ``grad_outputs`` should be a sequence of length matching ``output`` containing the "vector" in vector-Jacobian product... |
| `torch.autograd.gradcheck` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, inputs, eps, ...` | `<class 'bool'>` | Check gradients computed via small finite differences against analytical gradients wrt tensors in :attr:`inputs` that are of floating point or complex type and with ``requires_grad=True``. The chec... |
| `torch.autograd.gradgradcheck` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, inputs, grad_outputs, ...` | `<class 'bool'>` | Check gradients of gradients computed via small finite differences against analytical gradients wrt tensors in :attr:`inputs` and :attr:`grad_outputs` that are of floating point or complex type and... |
| `torch.autograd.handle_torch_function` | ❓ | ❓ | ❓ | ❓ | 🔴 | `public_api, relevant_args, args, ...` | `typing.Any` | Implement a function with checks for ``__torch_function__`` overrides. See torch::autograd::handle_torch_function for the equivalent of this function in the C++ implementation. Arguments --------- ... |
| `torch.autograd.has_torch_function` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Check for __torch_function__ implementations in the elements of an iterable or if a __torch_function__ mode is enabled. Considers exact ``Tensor`` s and ``Parameter`` s non-dispatchable. Use this t... |
| `torch.autograd.inference_mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mode` | `Any` | Context-manager that enables or disables inference mode. InferenceMode is a context manager analogous to :class:`~no_grad` to be used when you are certain your operations will have no interactions ... |
| `torch.autograd.is_multithreading_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Returns True if multithreading is currently enabled. |
| `torch.autograd.is_tensor_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `inp` | `Any` | Returns ``True`` if the passed-in input is a Tensor-like. Currently, this occurs whenever there's a ``__torch_function__`` attribute on the type of the input. Examples -------- A subclass of tensor... |
| `torch.autograd.is_view_replay_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Returns True if view-replay is currently enabled. |
| `torch.autograd.kineto_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | kineto_available() -> bool |
| `torch.autograd.no_grad` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Context-manager that disables gradient calculation. Disabling gradient calculation is useful for inference, when you are sure that you will not call :meth:`Tensor.backward()`. It will reduce memory... |
| `torch.autograd.set_detect_anomaly` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mode, check_nan` | `None` | Context-manager that sets the anomaly detection for the autograd engine on or off. ``set_detect_anomaly`` will enable or disable the autograd anomaly detection based on its argument :attr:`mode`. I... |
| `torch.autograd.set_grad_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mode` | `None` | Context-manager that sets gradient calculation on or off. ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`. It can be used as a context-manager or as a function.... |
| `torch.autograd.set_multithreading_enabled` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mode` | `None` | Context-manager that sets multithreaded backwards on or off. ``set_multithreading_enabled`` will enable or disable multithreaded backwards based on its argument :attr:`mode`. It can be used as a co... |
| `torch.autograd.variable` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` |  |
| | | | | | | | | |
| 🟦 BACKEND_MANAGEMENT | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.backends.ContextProp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `getter, setter` | `Any` |  |
| `torch.backends.PropModule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `m, name` | `Any` | Create a module object. The name must be a string; the optional doc argument can have any type. |
| `torch.backends.contextmanager` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` | @contextmanager decorator. Typical usage: @contextmanager def some_generator(<arguments>): <setup> try: yield <value> finally: <cleanup> This makes this: with some_generator(<arguments>) as <variab... |
| `torch.backends.disable_global_flags` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.backends.flags_frozen` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| | | | | | | | | |
| 🟦 COMPILATION | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.compiler.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch.compiler.ParamSpec` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, bound, covariant, ...` | `Any` | Parameter specification. |
| `torch.compiler.TypeVar` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, constraints, bound, ...` | `Any` | Type variable. Usage:: T = TypeVar('T') # Can be anything A = TypeVar('A', str, bytes) # Must be str or bytes Type variables exist primarily for the benefit of static type checkers. They serve as t... |
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
| `torch.cpu.AbstractContextManager` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | An abstract base class for context managers. |
| `torch.cpu.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch.cpu.Event` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cpu.Stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `priority` | `None` | N.B. This class only exists to facilitate device-agnostic code |
| `torch.cpu.StreamContext` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `Any` | Context-manager that selects a given stream. N.B. This class only exists to facilitate device-agnostic code |
| `torch.cpu.current_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'str'>` | Returns current device for cpu. Always 'cpu'. N.B. This function only exists to facilitate device-agnostic code |
| `torch.cpu.current_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.cpu.Stream'>` | Returns the currently selected :class:`Stream` for a given device. Args: device (torch.device or int, optional): Ignored. N.B. This function only exists to facilitate device-agnostic code |
| `torch.cpu.device_count` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Returns number of CPU devices (not cores). Always 1. N.B. This function only exists to facilitate device-agnostic code |
| `torch.cpu.is_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Returns a bool indicating if CPU is currently available. N.B. This function only exists to facilitate device-agnostic code |
| `torch.cpu.set_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Sets the current device, in CPU we do nothing. N.B. This function only exists to facilitate device-agnostic code |
| `torch.cpu.stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `<class 'contextlib.AbstractContextManager'>` | Wrapper around the Context-manager StreamContext that selects a given stream. N.B. This function only exists to facilitate device-agnostic code |
| `torch.cpu.synchronize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `None` | Waits for all kernels in all streams on the CPU device to complete. Args: device (torch.device or int, optional): ignored, there's only one CPU device. N.B. This function only exists to facilitate ... |
| | | | | | | | | |
| 🟦 CUDA_OPERATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.cuda.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch.cuda.BFloat16Storage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.cuda.BFloat16Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cuda.BoolStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.cuda.BoolTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cuda.ByteStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.cuda.ByteTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cuda.CUDAGraph` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Wrapper around a CUDA graph. .. warning:: This API is in beta and may change in future releases. |
| `torch.cuda.CUDAPluggableAllocator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `path_to_so_file, alloc_fn_name, free_fn_name` | `Any` | CUDA memory allocator loaded from a so file. |
| `torch.cuda.CharStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.cuda.CharTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cuda.ComplexDoubleStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.cuda.ComplexFloatStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.cuda.CudaError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `code` | `None` | Unspecified run-time error. |
| `torch.cuda.DeferredCudaCallError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Common base class for all non-exit exceptions. |
| `torch.cuda.DoubleStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.cuda.DoubleTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cuda.Event` | ❓ | ❓ | ❓ | ❓ | 🔴 | `enable_timing, blocking, interprocess` | `Any` | Wrapper around a CUDA event. CUDA events are synchronization markers that can be used to monitor the device's progress, to accurately measure timing, and to synchronize CUDA streams. The underlying... |
| `torch.cuda.ExternalStream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream_ptr, device, kwargs` | `Any` | Wrapper around an externally allocated CUDA stream. This class is used to wrap streams allocated in other libraries in order to facilitate data exchange and multi-library interactions. .. note:: Th... |
| `torch.cuda.FloatStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.cuda.FloatTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cuda.HalfStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.cuda.HalfTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cuda.IntStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.cuda.IntTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cuda.LongStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.cuda.LongTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cuda.MemPool` | ❓ | ❓ | ❓ | ❓ | 🔴 | `allocator` | `Any` | MemPool represents a pool of memory in a caching allocator. Currently, it's just the ID of the pool object maintained in the CUDACachingAllocator. Args: allocator(torch._C._cuda_CUDAAllocator, opti... |
| `torch.cuda.MemPoolContext` | ❓ | ❓ | ❓ | ❓ | 🔴 | `pool` | `Any` | MemPoolContext holds the currently active pool and stashes the previous pool. On deletion it makes the previous pool active. Args: pool(torch.cuda.MemPool): a MemPool object to be made active so th... |
| `torch.cuda.OutOfMemoryError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Exception raised when device is out of memory |
| `torch.cuda.ShortStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.cuda.ShortTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cuda.Stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device, priority, kwargs` | `Any` | Wrapper around a CUDA stream. A CUDA stream is a linear sequence of execution that belongs to a specific device, independent from other streams. It supports with statement as a context manager to e... |
| `torch.cuda.StreamContext` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `Any` | Context-manager that selects a given stream. All CUDA kernels queued within its context will be enqueued on a selected stream. Args: Stream (Stream): selected stream. This manager is a no-op if it'... |
| `torch.cuda.caching_allocator_alloc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size, device, stream` | `Any` | Perform a memory allocation using the CUDA memory allocator. Memory is allocated for a given device and a stream, this function is intended to be used for interoperability with other frameworks. Al... |
| `torch.cuda.caching_allocator_delete` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mem_ptr` | `Any` | Delete memory allocated using the CUDA memory allocator. Memory allocated with :func:`~torch.cuda.caching_allocator_alloc`. is freed here. The associated device and stream are tracked inside the al... |
| `torch.cuda.caching_allocator_enable` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `None` | Enable or disable the CUDA memory allocator. On by default. |
| `torch.cuda.can_device_access_peer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device, peer_device` | `<class 'bool'>` | Check if peer access between two devices is possible. |
| `torch.cuda.cast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typ, val` | `Any` | Cast a value to a type. This returns the value unchanged. To the type checker this signals that the return value has the designated type, but at runtime we intentionally don't check anything (we wa... |
| `torch.cuda.change_current_allocator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `allocator` | `None` | Change the currently used memory allocator to be the one provided. If the current allocator has already been used/initialized, this function will error. Args: allocator (torch.cuda.memory._CUDAAllo... |
| `torch.cuda.check_error` | ❓ | ❓ | ❓ | ❓ | 🔴 | `res` | `None` |  |
| `torch.cuda.classproperty` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` |  |
| `torch.cuda.clock_rate` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return the clock speed of the GPU SM in MHz (megahertz) over the past sample period as given by `nvidia-smi`. Args: device (torch.device or int, optional): selected device. Returns statistic for th... |
| `torch.cuda.cudaStatus` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.cuda.cudart` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Retrieves the CUDA runtime API module. This function initializes the CUDA runtime environment if it is not already initialized and returns the CUDA runtime API module (_cudart). The CUDA runtime AP... |
| `torch.cuda.current_blas_handle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Return cublasHandle_t pointer to current cuBLAS handle |
| `torch.cuda.current_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the index of a currently selected device. |
| `torch.cuda.current_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.cuda.streams.Stream'>` | Return the currently selected :class:`Stream` for a given device. Args: device (torch.device or int, optional): selected device. Returns the currently selected :class:`Stream` for the current devic... |
| `torch.cuda.default_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.cuda.streams.Stream'>` | Return the default :class:`Stream` for a given device. Args: device (torch.device or int, optional): selected device. Returns the default :class:`Stream` for the current device, given by :func:`~to... |
| `torch.cuda.device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `Any` | Context-manager that changes the selected device. Args: device (torch.device or int): device index to select. It's a no-op if this argument is a negative integer or ``None``. |
| `torch.cuda.device_count` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the number of GPUs available. .. note:: This API will NOT posion fork if NVML discovery succeeds. See :ref:`multiprocessing-poison-fork-note` for more details. |
| `torch.cuda.device_memory_used` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'int'>` | Return used global (device) memory in bytes as given by `nvidia-smi` or `amd-smi`. Args: device (torch.device or int, optional): selected device. Returns statistic for the current device, given by ... |
| `torch.cuda.device_of` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj` | `Any` | Context-manager that changes the current device to that of given object. You can use both tensors and storages as arguments. If a given object is not allocated on a GPU, this is a no-op. Args: obj ... |
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
| `torch.cuda.graph` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cuda_graph, pool, stream, ...` | `Any` | Context-manager that captures CUDA work into a :class:`torch.cuda.CUDAGraph` object for later replay. See :ref:`CUDA Graphs <cuda-graph-semantics>` for a general introduction, detailed use, and con... |
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
| `torch.distributed.AllToAllOptions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.distributed.AllreduceCoalescedOptions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.distributed.AllreduceOptions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.distributed.Backend` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name` | `Any` | An enum-like class for backends. Available backends: GLOO, NCCL, UCC, MPI, XCCL, and other registered backends. The values of this class are lowercase strings, e.g., ``"gloo"``. They can be accesse... |
| `torch.distributed.BackendConfig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `backend` | `Any` | Backend configuration class. |
| `torch.distributed.BarrierOptions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.distributed.BroadcastOptions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.distributed.BuiltinCommHookType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | An enum-like class for built-in communication hooks: ``ALLREDUCE`` and ``FP16_COMPRESS``. Members: ALLREDUCE FP16_COMPRESS |
| `torch.distributed.DebugLevel` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | An enum whose values correspond to different debug levels of the torch.distributed package. Currently supporting OFF, INFO, and DETAIL, which can be set via the TORCH_DISTRIBUTED_DEBUG environment ... |
| `torch.distributed.DeviceMesh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device_type, mesh, mesh_dim_names, ...` | `None` | DeviceMesh represents a mesh of devices, where layout of devices could be represented as a n-d dimension array, and each value of the n-d dimensional array is the global id of the default process g... |
| `torch.distributed.DistBackendError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Exception raised when a backend error occurs in distributed |
| `torch.distributed.DistError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Exception raised when an error occurs in the distributed library |
| `torch.distributed.DistNetworkError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Exception raised when a network error occurs in distributed |
| `torch.distributed.DistStoreError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Exception raised when an error occurs in the distributed store |
| `torch.distributed.FileStore` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | A store implementation that uses a file to store the underlying key-value pairs. Arguments: file_name (str): path of the file in which to store the key-value pairs world_size (int, optional): The t... |
| `torch.distributed.GatherOptions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.distributed.GradBucket` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | This class mainly passes a flattened gradient tensor (returned by :meth:`~torch.distributed.GradBucket.buffer`) to DDP communication hook. This tensor can be further decomposed into a list of per-p... |
| `torch.distributed.GroupMember` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Group member class. |
| `torch.distributed.HashStore` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | A thread-safe store implementation based on an underlying hashmap. This store can be used within the same process (for example, by other threads), but cannot be used across processes. Example:: >>>... |
| `torch.distributed.Logger` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.distributed.P2POp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `op, tensor, peer, ...` | `Any` | A class to build point-to-point operations for ``batch_isend_irecv``. This class builds the type of P2P operation, communication buffer, peer rank, Process Group, and tag. Instances of this class w... |
| `torch.distributed.PrefixStore` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | A wrapper around any of the 3 key-value stores (:class:`~torch.distributed.TCPStore`, :class:`~torch.distributed.FileStore`, and :class:`~torch.distributed.HashStore`) that adds a prefix to each ke... |
| `torch.distributed.ProcessGroup` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | A ProcessGroup is a communication primitive that allows for collective operations across a group of processes. This is a base class that provides the interface for all ProcessGroups. It is not mean... |
| `torch.distributed.ProcessGroupGloo` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.distributed.QueueEmptyError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Exception raised when an error occurs in the distributed store |
| `torch.distributed.ReduceOp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | An enum-like class for available reduction operations: ``SUM``, ``PRODUCT``, ``MIN``, ``MAX``, ``BAND``, ``BOR``, ``BXOR``, and ``PREMUL_SUM``. ``BAND``, ``BOR``, and ``BXOR`` reductions are not av... |
| `torch.distributed.ReduceOptions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.distributed.ReduceScatterOptions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.distributed.Reducer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.distributed.ScatterOptions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.distributed.Store` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Base class for all store implementations, such as the 3 provided by PyTorch distributed: (:class:`~torch.distributed.TCPStore`, :class:`~torch.distributed.FileStore`, and :class:`~torch.distributed... |
| `torch.distributed.TCPStore` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | A TCP-based distributed key-value store implementation. The server store holds the data, while the client stores can connect to the server store over TCP and perform actions such as :meth:`~torch.d... |
| `torch.distributed.Work` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | A `Work` object represents the handle to a pending asynchronous operation in PyTorch's distributed package. It is returned by non-blocking collective operations, such as `dist.all_reduce(tensor, as... |
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
| `torch.distributed.get_debug_level` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | get_debug_level() -> torch._C._distributed_c10d.DebugLevel Gets the debug level of the torch.distributed package. |
| `torch.distributed.get_default_backend_for_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'str'>` | Return the default backend for the given device. Args: Union[str, torch.device]: The device to get the default backend for. Returns: The default backend for the given device as a lower case string. |
| `torch.distributed.get_global_rank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group, group_rank` | `<class 'int'>` | Translate a group rank into a global rank. ``group_rank`` must be part of `group` otherwise this raises RuntimeError. Args: group (ProcessGroup): ProcessGroup to find the global rank from. group_ra... |
| `torch.distributed.get_group_rank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group, global_rank` | `<class 'int'>` | Translate a global rank into a group rank. ``global_rank`` must be part of ``group`` otherwise this raises RuntimeError. Args: group (ProcessGroup): ProcessGroup to find the relative rank. global_r... |
| `torch.distributed.get_node_local_rank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fallback_rank` | `<class 'int'>` | Return the local rank of the current process relative to the node. Semantically, this is a useful concept for mapping processes to devices. For example, on a node with 8 accelerator you could use t... |
| `torch.distributed.get_pg_count` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the number of process groups. |
| `torch.distributed.get_process_group_ranks` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group` | `list[int]` | Get all ranks associated with ``group``. Args: group (ProcessGroup): ProcessGroup to get all ranks from. Returns: List of global ranks ordered by group rank. |
| `torch.distributed.get_rank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group` | `<class 'int'>` | Return the rank of the current process in the provided ``group``, default otherwise. Rank is a unique identifier assigned to each process within a distributed process group. They are always consecu... |
| `torch.distributed.get_world_size` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group` | `<class 'int'>` | Return the number of processes in the current process group. Args: group (ProcessGroup, optional): The process group to work on. If None, the default process group will be used. Returns: The world ... |
| `torch.distributed.group` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Group class. Placeholder. |
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
| `torch.distributed.set_debug_level` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | set_debug_level(arg0: torch._C._distributed_c10d.DebugLevel) -> None Sets the debug level of the torch.distributed package. |
| `torch.distributed.set_debug_level_from_env` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | set_debug_level_from_env() -> None Sets the debug level of the torch.distributed package from the ``TORCH_DISTRIBUTED_DEBUG`` environment variable. |
| `torch.distributed.split_group` | ❓ | ❓ | ❓ | ❓ | 🔴 | `parent_pg, split_ranks, timeout, ...` | `typing.Optional[torch.distributed.distributed_c10d.ProcessGroup]` | Create a new process group splitted from the given parent process group. warning:: This is an experimental API and only the ``NCCL`` backend supports this API. Other backends will raise an error. U... |
| `torch.distributed.supports_complex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `reduceOp` | `<class 'bool'>` | Return true if reduce ops is supported. False otherwise. |
| | | | | | | | | |
| 🟦 DISTRIBUTIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.distributions.AbsTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cache_size` | `None` | Transform via the mapping :math:`y = |x|`. |
| `torch.distributions.AffineTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `loc, scale, event_dim, ...` | `None` | Transform via the pointwise affine mapping :math:`y = \text{loc} + \text{scale} \times x`. Args: loc (Tensor or float): Location parameter. scale (Tensor or float): Scale parameter. event_dim (int)... |
| `torch.distributions.Bernoulli` | ❓ | ❓ | ❓ | ❓ | 🔴 | `probs, logits, validate_args` | `None` | Creates a Bernoulli distribution parameterized by :attr:`probs` or :attr:`logits` (but not both). Samples are binary (0 or 1). They take the value `1` with probability `p` and `0` with probability ... |
| `torch.distributions.Beta` | ❓ | ❓ | ❓ | ❓ | 🔴 | `concentration1, concentration0, validate_args` | `None` | Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`. Example:: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5... |
| `torch.distributions.Binomial` | ❓ | ❓ | ❓ | ❓ | 🔴 | `total_count, probs, logits, ...` | `None` | Creates a Binomial distribution parameterized by :attr:`total_count` and either :attr:`probs` or :attr:`logits` (but not both). :attr:`total_count` must be broadcastable with :attr:`probs`/:attr:`l... |
| `torch.distributions.CatTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tseq, dim, lengths, ...` | `None` | Transform functor that applies a sequence of transforms `tseq` component-wise to each submatrix at `dim`, of length `lengths[dim]`, in a way compatible with :func:`torch.cat`. Example:: x0 = torch.... |
| `torch.distributions.Categorical` | ❓ | ❓ | ❓ | ❓ | 🔴 | `probs, logits, validate_args` | `None` | Creates a categorical distribution parameterized by either :attr:`probs` or :attr:`logits` (but not both). .. note:: It is equivalent to the distribution that :func:`torch.multinomial` samples from... |
| `torch.distributions.Cauchy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `loc, scale, validate_args` | `None` | Samples from a Cauchy (Lorentz) distribution. The distribution of the ratio of independent normally distributed random variables with means `0` follows a Cauchy distribution. Example:: >>> # xdocte... |
| `torch.distributions.Chi2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `df, validate_args` | `None` | Creates a Chi-squared distribution parameterized by shape parameter :attr:`df`. This is exactly equivalent to ``Gamma(alpha=0.5*df, beta=0.5)`` Example:: >>> # xdoctest: +IGNORE_WANT("non-determini... |
| `torch.distributions.ComposeTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `parts, cache_size` | `None` | Composes multiple transforms in a chain. The transforms being composed are responsible for caching. Args: parts (list of :class:`Transform`): A list of transforms to compose. cache_size (int): Size... |
| `torch.distributions.ContinuousBernoulli` | ❓ | ❓ | ❓ | ❓ | 🔴 | `probs, logits, lims, ...` | `None` | Creates a continuous Bernoulli distribution parameterized by :attr:`probs` or :attr:`logits` (but not both). The distribution is supported in [0, 1] and parameterized by 'probs' (in (0,1)) or 'logi... |
| `torch.distributions.CorrCholeskyTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cache_size` | `None` | Transforms an uncontrained real vector :math:`x` with length :math:`D*(D-1)/2` into the Cholesky factor of a D-dimension correlation matrix. This Cholesky factor is a lower triangular matrix with p... |
| `torch.distributions.CumulativeDistributionTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `distribution, cache_size` | `None` | Transform via the cumulative distribution function of a probability distribution. Args: distribution (Distribution): Distribution whose cumulative distribution function to use for the transformatio... |
| `torch.distributions.Dirichlet` | ❓ | ❓ | ❓ | ❓ | 🔴 | `concentration, validate_args` | `None` | Creates a Dirichlet distribution parameterized by concentration :attr:`concentration`. Example:: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = Dirichlet(torch.tensor([0.5, 0.5])) >>> m.... |
| `torch.distributions.Distribution` | ❓ | ❓ | ❓ | ❓ | 🔴 | `batch_shape, event_shape, validate_args` | `None` | Distribution is the abstract base class for probability distributions. |
| `torch.distributions.ExpTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cache_size` | `None` | Transform via the mapping :math:`y = \exp(x)`. |
| `torch.distributions.Exponential` | ❓ | ❓ | ❓ | ❓ | 🔴 | `rate, validate_args` | `None` | Creates a Exponential distribution parameterized by :attr:`rate`. Example:: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = Exponential(torch.tensor([1.0])) >>> m.sample() # Exponential d... |
| `torch.distributions.ExponentialFamily` | ❓ | ❓ | ❓ | ❓ | 🔴 | `batch_shape, event_shape, validate_args` | `None` | ExponentialFamily is the abstract base class for probability distributions belonging to an exponential family, whose probability mass/density function has the form is defined below .. math:: p_{F}(... |
| `torch.distributions.FisherSnedecor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `df1, df2, validate_args` | `None` | Creates a Fisher-Snedecor distribution parameterized by :attr:`df1` and :attr:`df2`. Example:: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = FisherSnedecor(torch.tensor([1.0]), torch.te... |
| `torch.distributions.Gamma` | ❓ | ❓ | ❓ | ❓ | 🔴 | `concentration, rate, validate_args` | `None` | Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`. Example:: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = Gamma(torch.tensor([1.0]), torch.tens... |
| `torch.distributions.GeneralizedPareto` | ❓ | ❓ | ❓ | ❓ | 🔴 | `loc, scale, concentration, ...` | `Any` | Creates a Generalized Pareto distribution parameterized by :attr:`loc`, :attr:`scale`, and :attr:`concentration`. The Generalized Pareto distribution is a family of continuous probability distribut... |
| `torch.distributions.Geometric` | ❓ | ❓ | ❓ | ❓ | 🔴 | `probs, logits, validate_args` | `None` | Creates a Geometric distribution parameterized by :attr:`probs`, where :attr:`probs` is the probability of success of Bernoulli trials. .. math:: P(X=k) = (1-p)^{k} p, k = 0, 1, ... .. note:: :func... |
| `torch.distributions.Gumbel` | ❓ | ❓ | ❓ | ❓ | 🔴 | `loc, scale, validate_args` | `None` | Samples from a Gumbel Distribution. Examples:: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0])) >>> m.sample() # sample from Gumbel distrib... |
| `torch.distributions.HalfCauchy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `scale, validate_args` | `None` | Creates a half-Cauchy distribution parameterized by `scale` where:: X ~ Cauchy(0, scale) Y = |X| ~ HalfCauchy(scale) Example:: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = HalfCauchy(t... |
| `torch.distributions.HalfNormal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `scale, validate_args` | `None` | Creates a half-normal distribution parameterized by `scale` where:: X ~ Normal(0, scale) Y = |X| ~ HalfNormal(scale) Example:: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = HalfNormal(t... |
| `torch.distributions.Independent` | ❓ | ❓ | ❓ | ❓ | 🔴 | `base_distribution, reinterpreted_batch_ndims, validate_args` | `None` | Reinterprets some of the batch dims of a distribution as event dims. This is mainly useful for changing the shape of the result of :meth:`log_prob`. For example to create a diagonal Normal distribu... |
| `torch.distributions.IndependentTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `base_transform, reinterpreted_batch_ndims, cache_size` | `None` | Wrapper around another transform to treat ``reinterpreted_batch_ndims``-many extra of the right most dimensions as dependent. This has no effect on the forward or backward transforms, but does sum ... |
| `torch.distributions.InverseGamma` | ❓ | ❓ | ❓ | ❓ | 🔴 | `concentration, rate, validate_args` | `None` | Creates an inverse gamma distribution parameterized by :attr:`concentration` and :attr:`rate` where:: X ~ Gamma(concentration, rate) Y = 1 / X ~ InverseGamma(concentration, rate) Example:: >>> # xd... |
| `torch.distributions.Kumaraswamy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `concentration1, concentration0, validate_args` | `None` | Samples from a Kumaraswamy distribution. Example:: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = Kumaraswamy(torch.tensor([1.0]), torch.tensor([1.0])) >>> m.sample() # sample from a Kum... |
| `torch.distributions.LKJCholesky` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dim, concentration, validate_args` | `None` | LKJ distribution for lower Cholesky factor of correlation matrices. The distribution is controlled by ``concentration`` parameter :math:`\eta` to make the probability of the correlation matrix :mat... |
| `torch.distributions.Laplace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `loc, scale, validate_args` | `None` | Creates a Laplace distribution parameterized by :attr:`loc` and :attr:`scale`. Example:: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = Laplace(torch.tensor([0.0]), torch.tensor([1.0])) ... |
| `torch.distributions.LogNormal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `loc, scale, validate_args` | `None` | Creates a log-normal distribution parameterized by :attr:`loc` and :attr:`scale` where:: X ~ Normal(loc, scale) Y = exp(X) ~ LogNormal(loc, scale) Example:: >>> # xdoctest: +IGNORE_WANT("non-determ... |
| `torch.distributions.LogisticNormal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `loc, scale, validate_args` | `None` | Creates a logistic-normal distribution parameterized by :attr:`loc` and :attr:`scale` that define the base `Normal` distribution transformed with the `StickBreakingTransform` such that:: X ~ Logist... |
| `torch.distributions.LowRankMultivariateNormal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `loc, cov_factor, cov_diag, ...` | `None` | Creates a multivariate normal distribution with covariance matrix having a low-rank form parameterized by :attr:`cov_factor` and :attr:`cov_diag`:: covariance_matrix = cov_factor @ cov_factor.T + c... |
| `torch.distributions.LowerCholeskyTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cache_size` | `None` | Transform from unconstrained matrices to lower-triangular matrices with nonnegative diagonal entries. This is useful for parameterizing positive definite matrices in terms of their Cholesky factori... |
| `torch.distributions.MixtureSameFamily` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mixture_distribution, component_distribution, validate_args` | `None` | The `MixtureSameFamily` distribution implements a (batch of) mixture distribution where all component are from different parameterizations of the same distribution type. It is parameterized by a `C... |
| `torch.distributions.Multinomial` | ❓ | ❓ | ❓ | ❓ | 🔴 | `total_count, probs, logits, ...` | `None` | Creates a Multinomial distribution parameterized by :attr:`total_count` and either :attr:`probs` or :attr:`logits` (but not both). The innermost dimension of :attr:`probs` indexes over categories. ... |
| `torch.distributions.MultivariateNormal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `loc, covariance_matrix, precision_matrix, ...` | `None` | Creates a multivariate normal (also called Gaussian) distribution parameterized by a mean vector and a covariance matrix. The multivariate normal distribution can be parameterized either in terms o... |
| `torch.distributions.NegativeBinomial` | ❓ | ❓ | ❓ | ❓ | 🔴 | `total_count, probs, logits, ...` | `None` | Creates a Negative Binomial distribution, i.e. distribution of the number of successful independent and identical Bernoulli trials before :attr:`total_count` failures are achieved. The probability ... |
| `torch.distributions.Normal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `loc, scale, validate_args` | `None` | Creates a normal (also called Gaussian) distribution parameterized by :attr:`loc` and :attr:`scale`. Example:: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = Normal(torch.tensor([0.0]), ... |
| `torch.distributions.OneHotCategorical` | ❓ | ❓ | ❓ | ❓ | 🔴 | `probs, logits, validate_args` | `None` | Creates a one-hot categorical distribution parameterized by :attr:`probs` or :attr:`logits`. Samples are one-hot coded vectors of size ``probs.size(-1)``. .. note:: The `probs` argument must be non... |
| `torch.distributions.OneHotCategoricalStraightThrough` | ❓ | ❓ | ❓ | ❓ | 🔴 | `probs, logits, validate_args` | `None` | Creates a reparameterizable :class:`OneHotCategorical` distribution based on the straight- through gradient estimator from [1]. [1] Estimating or Propagating Gradients Through Stochastic Neurons fo... |
| `torch.distributions.Pareto` | ❓ | ❓ | ❓ | ❓ | 🔴 | `scale, alpha, validate_args` | `None` | Samples from a Pareto Type 1 distribution. Example:: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0])) >>> m.sample() # sample from a Pareto... |
| `torch.distributions.Poisson` | ❓ | ❓ | ❓ | ❓ | 🔴 | `rate, validate_args` | `None` | Creates a Poisson distribution parameterized by :attr:`rate`, the rate parameter. Samples are nonnegative integers, with a pmf given by .. math:: \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!} Examp... |
| `torch.distributions.PositiveDefiniteTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cache_size` | `None` | Transform from unconstrained matrices to positive-definite matrices. |
| `torch.distributions.PowerTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `exponent, cache_size` | `None` | Transform via the mapping :math:`y = x^{\text{exponent}}`. |
| `torch.distributions.RelaxedBernoulli` | ❓ | ❓ | ❓ | ❓ | 🔴 | `temperature, probs, logits, ...` | `None` | Creates a RelaxedBernoulli distribution, parametrized by :attr:`temperature`, and either :attr:`probs` or :attr:`logits` (but not both). This is a relaxed version of the `Bernoulli` distribution, s... |
| `torch.distributions.RelaxedOneHotCategorical` | ❓ | ❓ | ❓ | ❓ | 🔴 | `temperature, probs, logits, ...` | `None` | Creates a RelaxedOneHotCategorical distribution parametrized by :attr:`temperature`, and either :attr:`probs` or :attr:`logits`. This is a relaxed version of the :class:`OneHotCategorical` distribu... |
| `torch.distributions.ReshapeTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `in_shape, out_shape, cache_size` | `None` | Unit Jacobian transform to reshape the rightmost part of a tensor. Note that ``in_shape`` and ``out_shape`` must have the same number of elements, just as for :meth:`torch.Tensor.reshape`. Argument... |
| `torch.distributions.SigmoidTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cache_size` | `None` | Transform via the mapping :math:`y = \frac{1}{1 + \exp(-x)}` and :math:`x = \text{logit}(y)`. |
| `torch.distributions.SoftmaxTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cache_size` | `None` | Transform from unconstrained space to the simplex via :math:`y = \exp(x)` then normalizing. This is not bijective and cannot be used for HMC. However this acts mostly coordinate-wise (except for th... |
| `torch.distributions.SoftplusTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cache_size` | `None` | Transform via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`. The implementation reverts to the linear function when :math:`x > 20`. |
| `torch.distributions.StackTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tseq, dim, cache_size` | `None` | Transform functor that applies a sequence of transforms `tseq` component-wise to each submatrix at `dim` in a way compatible with :func:`torch.stack`. Example:: x = torch.stack([torch.range(1, 10),... |
| `torch.distributions.StickBreakingTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cache_size` | `None` | Transform from unconstrained space to the simplex of one additional dimension via a stick-breaking process. This transform arises as an iterated sigmoid transform in a stick-breaking construction o... |
| `torch.distributions.StudentT` | ❓ | ❓ | ❓ | ❓ | 🔴 | `df, loc, scale, ...` | `None` | Creates a Student's t-distribution parameterized by degree of freedom :attr:`df`, mean :attr:`loc` and scale :attr:`scale`. Example:: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = Stude... |
| `torch.distributions.TanhTransform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cache_size` | `None` | Transform via the mapping :math:`y = \tanh(x)`. It is equivalent to .. code-block:: python ComposeTransform( [ AffineTransform(0.0, 2.0), SigmoidTransform(), AffineTransform(-1.0, 2.0), ] ) However... |
| `torch.distributions.Transform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cache_size` | `None` | Abstract class for invertable transformations with computable log det jacobians. They are primarily used in :class:`torch.distributions.TransformedDistribution`. Caching is useful for transforms wh... |
| `torch.distributions.TransformedDistribution` | ❓ | ❓ | ❓ | ❓ | 🔴 | `base_distribution, transforms, validate_args` | `None` | Extension of the Distribution class, which applies a sequence of Transforms to a base distribution. Let f be the composition of transforms applied:: X ~ BaseDistribution Y = f(X) ~ TransformedDistr... |
| `torch.distributions.Uniform` | ❓ | ❓ | ❓ | ❓ | 🔴 | `low, high, validate_args` | `None` | Generates uniformly distributed random samples from the half-open interval ``[low, high)``. Example:: >>> m = Uniform(torch.tensor([0.0]), torch.tensor([5.0])) >>> m.sample() # uniformly distribute... |
| `torch.distributions.VonMises` | ❓ | ❓ | ❓ | ❓ | 🔴 | `loc, concentration, validate_args` | `None` | A circular von Mises distribution. This implementation uses polar coordinates. The ``loc`` and ``value`` args can be any real number (to facilitate unconstrained optimization), but are interpreted ... |
| `torch.distributions.Weibull` | ❓ | ❓ | ❓ | ❓ | 🔴 | `scale, concentration, validate_args` | `None` | Samples from a two-parameter Weibull distribution. Example: >>> # xdoctest: +IGNORE_WANT("non-deterministic") >>> m = Weibull(torch.tensor([1.0]), torch.tensor([1.0])) >>> m.sample() # sample from ... |
| `torch.distributions.Wishart` | ❓ | ❓ | ❓ | ❓ | 🔴 | `df, covariance_matrix, precision_matrix, ...` | `None` | Creates a Wishart distribution parameterized by a symmetric positive definite matrix :math:`\Sigma`, or its Cholesky decomposition :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top` Example: >>> #... |
| `torch.distributions.kl_divergence` | ❓ | ❓ | ❓ | ❓ | 🔴 | `p, q` | `<class 'torch.Tensor'>` | Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions. .. math:: KL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)} \,dx Args: p (Distribution): A :class:`~torch.distribution... |
| `torch.distributions.register_kl` | ❓ | ❓ | ❓ | ❓ | 🔴 | `type_p, type_q` | `Any` | Decorator to register a pairwise function with :meth:`kl_divergence`. Usage:: @register_kl(Normal, Normal) def kl_normal_normal(p, q): # insert implementation here Lookup returns the most specific ... |
| | | | | | | | | |
| 🟦 MODEL_EXPORT | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.export.AdditionalInputs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Infers dynamic_shapes based on additional inputs. This is useful particularly for deployment engineers who, on the one hand, may have access to ample testing or profiling data that can provide a fa... |
| `torch.export.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch.export.CustomDecompTable` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | This is a custom dictionary that is specifically used for handling decomp_table in export. The reason we need this is because in the new world, you can only *delete* an op from decomp table to pres... |
| `torch.export.Dim` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, min, max` | `Any` | :func:`Dim` constructs a type analogous to a named symbolic integer with a range. It can be used to describe multiple possible values of a dynamic tensor dimension. Note that different dynamic dime... |
| `torch.export.Enum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ... RED = 1 ... BLUE = 2 ... GREEN = 3 Access them by: - attribute access:: >>> Color.RED <Color.RED: 1> - value... |
| `torch.export.ExportBackwardSignature` | ❓ | ❓ | ❓ | ❓ | 🔴 | `gradients_to_parameters, gradients_to_user_inputs, loss_output` | `None` | ExportBackwardSignature(gradients_to_parameters: dict[str, str], gradients_to_user_inputs: dict[str, str], loss_output: str) |
| `torch.export.ExportGraphSignature` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input_specs, output_specs` | `None` | :class:`ExportGraphSignature` models the input/output signature of Export Graph, which is a fx.Graph with stronger invariants gurantees. Export Graph is functional and does not access "states" like... |
| `torch.export.ExportedProgram` | ❓ | ❓ | ❓ | ❓ | 🔴 | `root, graph, graph_signature, ...` | `Any` | Package of a program from :func:`export`. It contains an :class:`torch.fx.Graph` that represents Tensor computation, a state_dict containing tensor values of all lifted parameters and buffers, and ... |
| `torch.export.FlatArgsAdapter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Adapts input arguments with ``input_spec`` to align ``target_spec``. |
| `torch.export.Iterator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.export.ModuleCallEntry` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fqn, signature` | `None` | ModuleCallEntry(fqn: str, signature: Optional[torch.export.exported_program.ModuleCallSignature] = None) |
| `torch.export.ModuleCallSignature` | ❓ | ❓ | ❓ | ❓ | 🔴 | `inputs, outputs, in_spec, ...` | `None` | ModuleCallSignature(inputs: list[typing.Union[torch.export.graph_signature.TensorArgument, torch.export.graph_signature.SymIntArgument, torch.export.graph_signature.SymFloatArgument, torch.export.g... |
| `torch.export.PassManager` | ❓ | ❓ | ❓ | ❓ | 🔴 | `passes, constraints, steps, ...` | `Any` | Construct a PassManager. Collects passes and constraints. This defines the pass schedule, manages pass constraints and pass execution. Args: passes (Optional[List[Callable]]): List of passes. A pas... |
| `torch.export.PassResult` | ❓ | ❓ | ❓ | ❓ | 🔴 | `graph_module, modified` | `Any` | Result of a pass: graph_module: The modified graph module modified: A flag for if the pass has modified the graph module .. warning:: This API is experimental and is *NOT* backward-compatible. |
| `torch.export.ShapesCollection` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Builder for dynamic_shapes. Used to assign dynamic shape specifications to tensors that appear in inputs. This is useful particularly when :func:`args` is a nested input structure, and it's easier ... |
| `torch.export.UnflattenedModule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `export_module, flat_args_adapter` | `Any` | Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing them to be nested in a tree structure. You can assign the su... |
| `torch.export.auto` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value` | `Any` | Instances are replaced with an appropriate value in Enum class suites. |
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
| 🟦 SIGNAL_PROCESSING | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.fft.Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.fft.fft` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | fft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor Computes the one dimensional discrete Fourier transform of :attr:`input`. Note: The Fourier domain representation of any real signal sat... |
| `torch.fft.fft2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | fft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor Computes the 2 dimensional discrete Fourier transform of :attr:`input`. Equivalent to :func:`~torch.fft.fftn` but FFTs only the l... |
| `torch.fft.fftfreq` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | fftfreq(n, d=1.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Computes the discrete Fourier Transform sample frequencies for a signal of size :attr:`n... |
| `torch.fft.fftn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | fftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor Computes the N dimensional discrete Fourier transform of :attr:`input`. Note: The Fourier domain representation of any real signal sa... |
| `torch.fft.fftshift` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | fftshift(input, dim=None) -> Tensor Reorders n-dimensional FFT data, as provided by :func:`~torch.fft.fftn`, to have negative frequency terms first. This performs a periodic shift of n-dimensional ... |
| `torch.fft.hfft` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor Computes the one dimensional discrete Fourier transform of a Hermitian symmetric :attr:`input` signal. Note: :func:`~torch.fft.hfft`/:f... |
| `torch.fft.hfft2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor Computes the 2-dimensional discrete Fourier transform of a Hermitian symmetric :attr:`input` signal. Equivalent to :func:`~torch... |
| `torch.fft.hfftn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor Computes the n-dimensional discrete Fourier transform of a Hermitian symmetric :attr:`input` signal. :attr:`input` is interpreted as... |
| `torch.fft.ifft` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ifft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor Computes the one dimensional inverse discrete Fourier transform of :attr:`input`. Note: Supports torch.half and torch.chalf on CUDA wit... |
| `torch.fft.ifft2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ifft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor Computes the 2 dimensional inverse discrete Fourier transform of :attr:`input`. Equivalent to :func:`~torch.fft.ifftn` but IFFTs... |
| `torch.fft.ifftn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ifftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor Computes the N dimensional inverse discrete Fourier transform of :attr:`input`. Note: Supports torch.half and torch.chalf on CUDA wi... |
| `torch.fft.ifftshift` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ifftshift(input, dim=None) -> Tensor Inverse of :func:`~torch.fft.fftshift`. Args: input (Tensor): the tensor in FFT order dim (int, Tuple[int], optional): The dimensions to rearrange. Only dimensi... |
| `torch.fft.ihfft` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ihfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor Computes the inverse of :func:`~torch.fft.hfft`. :attr:`input` must be a real-valued signal, interpreted in the Fourier domain. The IF... |
| `torch.fft.ihfft2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ihfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor Computes the 2-dimensional inverse discrete Fourier transform of real :attr:`input`. Equivalent to :func:`~torch.fft.ihfftn` bu... |
| `torch.fft.ihfftn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ihfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor Computes the N-dimensional inverse discrete Fourier transform of real :attr:`input`. :attr:`input` must be a real-valued signal, in... |
| `torch.fft.irfft` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | irfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor Computes the inverse of :func:`~torch.fft.rfft`. :attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier domain, a... |
| `torch.fft.irfft2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | irfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor Computes the inverse of :func:`~torch.fft.rfft2`. Equivalent to :func:`~torch.fft.irfftn` but IFFTs only the last two dimension... |
| `torch.fft.irfftn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | irfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor Computes the inverse of :func:`~torch.fft.rfftn`. :attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier domai... |
| `torch.fft.rfft` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | rfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor Computes the one dimensional Fourier transform of real-valued :attr:`input`. The FFT of a real signal is Hermitian-symmetric, ``X[i] = ... |
| `torch.fft.rfft2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | rfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor Computes the 2-dimensional discrete Fourier transform of real :attr:`input`. Equivalent to :func:`~torch.fft.rfftn` but FFTs onl... |
| `torch.fft.rfftfreq` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | rfftfreq(n, d=1.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor Computes the sample frequencies for :func:`~torch.fft.rfft` with a signal of size :attr... |
| `torch.fft.rfftn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | rfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor Computes the N-dimensional discrete Fourier transform of real :attr:`input`. The FFT of a real signal is Hermitian-symmetric, ``X[i_... |
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
| `torch.futures.Future` | ❓ | ❓ | ❓ | ❓ | 🔴 | `devices` | `Any` | Wrapper around a ``torch._C.Future`` which encapsulates an asynchronous execution of a callable, e.g. :meth:`~torch.distributed.rpc.rpc_async`. It also exposes a set of APIs to add callback functio... |
| `torch.futures.Generic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Abstract base class for generic types. A generic type is typically declared by inheriting from this class parameterized with one or more type variables. For example, a generic mapping type might be... |
| `torch.futures.TypeVar` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, constraints, bound, ...` | `Any` | Type variable. Usage:: T = TypeVar('T') # Can be anything A = TypeVar('A', str, bytes) # Must be str or bytes Type variables exist primarily for the benefit of static type checkers. They serve as t... |
| `torch.futures.cast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typ, val` | `Any` | Cast a value to a type. This returns the value unchanged. To the type checker this signals that the return value has the designated type, but at runtime we intentionally don't check anything (we wa... |
| `torch.futures.collect_all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `futures` | `Future[list[Future]]` | Collects the provided :class:`~torch.futures.Future` objects into a single combined :class:`~torch.futures.Future` that is completed when all of the sub-futures are completed. Args: futures (list):... |
| `torch.futures.wait_all` | ❓ | ❓ | ❓ | ❓ | 🔴 | `futures` | `list` | Waits for all provided futures to be complete, and returns the list of completed values. If any of the futures encounters an error, the method will exit early and report the error not waiting for o... |
| | | | | | | | | |
| 🟦 GRAPH_TRANSFORMATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.fx.CodeGen` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | .. warning:: This API is experimental and is *NOT* backward-compatible. |
| `torch.fx.Graph` | ❓ | ❓ | ❓ | ❓ | 🔴 | `owning_module, tracer_cls, tracer_extras` | `Any` | ``Graph`` is the main data structure used in the FX Intermediate Representation. It consists of a series of ``Node`` s, each representing callsites (or other syntactic constructs). The list of ``No... |
| `torch.fx.GraphModule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | GraphModule is an nn.Module generated from an fx.Graph. Graphmodule has a ``graph`` attribute, as well as ``code`` and ``forward`` attributes generated from that ``graph``. .. warning:: When ``grap... |
| `torch.fx.Interpreter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module, garbage_collect_values, graph` | `Any` | An Interpreter executes an FX graph Node-by-Node. This pattern can be useful for many things, including writing code transformations as well as analysis passes. Methods in the Interpreter class can... |
| `torch.fx.Node` | ❓ | ❓ | ❓ | ❓ | 🔴 | `graph, name, op, ...` | `None` | ``Node`` is the data structure that represents individual operations within a ``Graph``. For the most part, Nodes represent callsites to various entities, such as operators, methods, and Modules (s... |
| `torch.fx.Proxy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `node, tracer` | `Any` | ``Proxy`` objects are ``Node`` wrappers that flow through the program during symbolic tracing and record all the operations (``torch`` function calls, method calls, operators) that they touch into ... |
| `torch.fx.ProxyableClassMeta` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, bases, attrs` | `Any` | ProxyableClassMeta allows you to make construction of a given Python class symbolically traceable. For example:: import torch import torch.fx class TensorPair(metaclass=torch.fx.ProxyableClassMeta)... |
| `torch.fx.Tracer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `autowrap_modules, autowrap_functions, param_shapes_constant` | `None` | Tracer(autowrap_modules=(math,), autowrap_functions=()) ``Tracer`` is the class that implements the symbolic tracing functionality of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is... |
| `torch.fx.Transformer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module` | `Any` | ``Transformer`` is a special type of interpreter that produces a new ``Module``. It exposes a ``transform()`` method that returns the transformed ``Module``. ``Transformer`` does not require argume... |
| `torch.fx.has_side_effect` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `typing.Callable[~_P, ~_R]` | .. warning:: This API is experimental and is *NOT* backward-compatible. |
| `torch.fx.map_arg` | ❓ | ❓ | ❓ | ❓ | 🔴 | `a, fn` | `~ArgumentT` | Apply fn recursively to each Node appearing in arg. arg may be a list, tuple, slice, or dict with string keys: the return value will have the same type and structure. .. note:: Backwards-compatibil... |
| `torch.fx.replace_pattern` | ❓ | ❓ | ❓ | ❓ | 🔴 | `gm, pattern, replacement` | `list[torch.fx.subgraph_rewriter.Match]` | Matches all possible non-overlapping sets of operators and their data dependencies (``pattern``) in the Graph of a GraphModule (``gm``), then replaces each of these matched subgraphs with another s... |
| `torch.fx.symbolic_trace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `root, concrete_args` | `<class 'torch.fx.graph_module.GraphModule'>` | Symbolic tracing API Given an ``nn.Module`` or function instance ``root``, this function will return a ``GraphModule`` constructed by recording operations seen while tracing through ``root``. ``con... |
| `torch.fx.wrap` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn_or_name` | `Any` | This function can be called at module-level scope to register fn_or_name as a "leaf function". A "leaf function" will be preserved as a CallFunction node in the FX trace instead of being traced thr... |
| | | | | | | | | |
| 🟦 MODEL_HUB | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.hub.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch.hub.HTTPError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `url, code, msg, ...` | `Any` | Raised when HTTP error occurs, but also acts like non-error return |
| `torch.hub.Path` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | PurePath subclass that can make system calls. Path represents a filesystem path but unlike PurePath, also offers methods to do system calls on path objects. Depending on your system, instantiating ... |
| `torch.hub.Request` | ❓ | ❓ | ❓ | ❓ | 🔴 | `url, data, headers, ...` | `Any` |  |
| `torch.hub.URLError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `reason, filename` | `Any` | Base class for I/O related errors. |
| `torch.hub.deprecated` | ❓ | ❓ | ❓ | ❓ | 🔴 | `message, category, stacklevel` | `None` | Indicate that a class, function or overload is deprecated. When this decorator is applied to an object, the type checker will generate a diagnostic on usage of the deprecated object. Usage: @deprec... |
| `torch.hub.download_url_to_file` | ❓ | ❓ | ❓ | ❓ | 🔴 | `url, dst, hash_prefix, ...` | `None` | Download object at the given URL to a local path. Args: url (str): URL of the object to download dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file`` hash_prefix (str, opti... |
| `torch.hub.get_dir` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'str'>` | Get the Torch Hub cache directory used for storing downloaded models & weights. If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where environment variable ``$TORCH_... |
| `torch.hub.help` | ❓ | ❓ | ❓ | ❓ | 🔴 | `github, model, force_reload, ...` | `Any` | Show the docstring of entrypoint ``model``. Args: github (str): a string with format <repo_owner/repo_name[:ref]> with an optional ref (a tag or a branch). If ``ref`` is not specified, the default ... |
| `torch.hub.list` | ❓ | ❓ | ❓ | ❓ | 🔴 | `github, force_reload, skip_validation, ...` | `Any` | List all callable entrypoints available in the repo specified by ``github``. Args: github (str): a string with format "repo_owner/repo_name[:ref]" with an optional ref (tag or branch). If ``ref`` i... |
| `torch.hub.load` | ❓ | ❓ | ❓ | ❓ | 🔴 | `repo_or_dir, model, args, ...` | `Any` | Load a model from a github repo or a local directory. Note: Loading a model is the typical use case, but this can also be used to for loading other objects such as tokenizers, loss functions, etc. ... |
| `torch.hub.load_state_dict_from_url` | ❓ | ❓ | ❓ | ❓ | 🔴 | `url, model_dir, map_location, ...` | `dict[str, typing.Any]` | Loads the Torch serialized object at the given URL. If downloaded file is a zip file, it will be automatically decompressed. If the object is already present in `model_dir`, it's deserialized and r... |
| `torch.hub.set_dir` | ❓ | ❓ | ❓ | ❓ | 🔴 | `d` | `None` | Optionally set the Torch Hub directory used to save downloaded models & weights. Args: d (str): path to a local folder to save downloaded models & weights. |
| `torch.hub.tqdm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `_, __` | `Any` | Decorate an iterable object, returning an iterator which acts exactly like the original iterable, but prints a dynamically updating progressbar every time a value is requested. Parameters ---------... |
| `torch.hub.urlopen` | ❓ | ❓ | ❓ | ❓ | 🔴 | `url, data, timeout, ...` | `Any` | Open the URL url, which can be either a string or a Request object. *data* must be an object specifying additional data to be sent to the server, or None if no such data is needed. See Request for ... |
| `torch.hub.urlparse` | ❓ | ❓ | ❓ | ❓ | 🔴 | `url, scheme, allow_fragments` | `Any` | Parse a URL into 6 components: <scheme>://<netloc>/<path>;<params>?<query>#<fragment> The result is a named 6-tuple with fields corresponding to the above. It is either a ParseResult or ParseResult... |
| | | | | | | | | |
| 🟦 JIT_COMPILATION | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.jit.Attribute` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, type` | `Any` | This method is a pass-through function that returns `value`, mostly used to indicate to the TorchScript compiler that the left-hand side expression is a class instance attribute with type of `type`... |
| `torch.jit.CompilationUnit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.jit.Error` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Common base class for all non-exit exceptions. |
| `torch.jit.Future` | ❓ | ❓ | ❓ | ❓ | 🔴 | `devices` | `Any` | Wrapper around a ``torch._C.Future`` which encapsulates an asynchronous execution of a callable, e.g. :meth:`~torch.distributed.rpc.rpc_async`. It also exposes a set of APIs to add callback functio... |
| `torch.jit.Iterator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.jit.ONNXTracedModule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `inner, strict, force_outplace, ...` | `Any` | Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing them to be nested in a tree structure. You can assign the su... |
| `torch.jit.RecursiveScriptClass` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cpp_class` | `Any` | Wrapper for a TorchScript class instance for use in Python. An analogue of RecursiveScriptModule for regular objects that are not modules. This class is a wrapper around a torch._C.ScriptObject tha... |
| `torch.jit.RecursiveScriptModule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cpp_module` | `Any` | Retain the existing isinstance(ScriptModule) behavior. The core data structure in TorchScript is the ``ScriptModule``. It is an analogue of torch's ``nn.Module`` and represents an entire model as a... |
| `torch.jit.ScriptFunction` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Functionally equivalent to a :class:`ScriptModule`, but represents a single function and does not have any attributes or Parameters. |
| `torch.jit.ScriptModule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Wrapper for C++ torch::jit::Module with methods, attributes, and parameters. A wrapper around C++ ``torch::jit::Module``. ``ScriptModule``\s contain methods, attributes, parameters, and constants. ... |
| `torch.jit.ScriptWarning` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Base class for warning categories. |
| `torch.jit.TopLevelTracedModule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `orig, id_set, _compilation_unit` | `Any` | Wrapper for C++ torch::jit::Module with methods, attributes, and parameters. A wrapper around C++ ``torch::jit::Module``. ``ScriptModule``\s contain methods, attributes, parameters, and constants. ... |
| `torch.jit.TracedModule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `orig, id_set, _compilation_unit` | `Any` | Wrapper for C++ torch::jit::Module with methods, attributes, and parameters. A wrapper around C++ ``torch::jit::Module``. ``ScriptModule``\s contain methods, attributes, parameters, and constants. ... |
| `torch.jit.TracerWarning` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Base class for warning categories. |
| `torch.jit.TracingCheckError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `graph_diff_error, tensor_compare_error, extra_msg` | `Any` | Common base class for all non-exit exceptions. |
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
| `torch.jit.last_executed_optimized_graph` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | _last_executed_optimized_graph() -> torch._C.Graph Retrieve the optimized graph that was run the last time the graph executor ran on this thread |
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
| `torch.jit.strict_fusion` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Give errors if not all nodes have been fused in inference, or symbolically differentiated in training. Example: Forcing fusion of additions. .. code-block:: python @torch.jit.script def foo(x): wit... |
| `torch.jit.trace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func, example_inputs, optimize, ...` | `Any` | Trace a function and return an executable or :class:`ScriptFunction` that will be optimized using just-in-time compilation. Tracing is ideal for code that operates only on ``Tensor``\\s and lists, ... |
| `torch.jit.trace_module` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mod, inputs, optimize, ...` | `Any` | Trace a module and return an executable :class:`ScriptModule` that will be optimized using just-in-time compilation. When a module is passed to :func:`torch.jit.trace <torch.jit.trace>`, only the `... |
| `torch.jit.unused` | ❓ | ❓ | ❓ | ❓ | 🔴 | `fn` | `Any` | This decorator indicates to the compiler that a function or method should be ignored and replaced with the raising of an exception. This allows you to leave code in your model that is not yet Torch... |
| `torch.jit.wait` | ❓ | ❓ | ❓ | ❓ | 🔴 | `future` | `Any` | Force completion of a `torch.jit.Future[T]` asynchronous task, returning the result of the task. See :func:`~fork` for docs and examples. Args: future (torch.jit.Future[T]): an asynchronous task re... |
| | | | | | | | | |
| 🟦 LINEAR_ALGEBRA | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.linalg.LinAlgError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Error raised by torch.linalg function when the cause of error is a numerical inconsistency in the data. For example, you can the torch.linalg.inv function will raise torch.linalg.LinAlgError when i... |
| `torch.linalg.cholesky` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.cholesky(A, *, upper=False, out=None) -> Tensor Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix. Letting :math:`\mathbb{K}` be :math:`\m... |
| `torch.linalg.cholesky_ex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.cholesky_ex(A, *, upper=False, check_errors=False, out=None) -> (Tensor, Tensor) Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix. This f... |
| `torch.linalg.cond` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.cond(A, p=None, *, out=None) -> Tensor Computes the condition number of a matrix with respect to a matrix norm. Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, the **... |
| `torch.linalg.cross` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.cross(input, other, *, dim=-1, out=None) -> Tensor Computes the cross product of two 3-dimensional vectors. Supports input of float, double, cfloat and cdouble dtypes. Also supports batches ... |
| `torch.linalg.det` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.det(A, *, out=None) -> Tensor Computes the determinant of a square matrix. Supports input of float, double, cfloat and cdouble dtypes. Also supports batches of matrices, and if :attr:`A` is ... |
| `torch.linalg.diagonal` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.diagonal(A, *, offset=0, dim1=-2, dim2=-1) -> Tensor Alias for :func:`torch.diagonal` with defaults :attr:`dim1`\ `= -2`, :attr:`dim2`\ `= -1`. |
| `torch.linalg.eig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.eig(A, *, out=None) -> (Tensor, Tensor) Computes the eigenvalue decomposition of a square matrix if it exists. Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, the **e... |
| `torch.linalg.eigh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.eigh(A, UPLO='L', *, out=None) -> (Tensor, Tensor) Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix. Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or ... |
| `torch.linalg.eigvals` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.eigvals(A, *, out=None) -> Tensor Computes the eigenvalues of a square matrix. Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, the **eigenvalues** of a square matrix ... |
| `torch.linalg.eigvalsh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.eigvalsh(A, UPLO='L', *, out=None) -> Tensor Computes the eigenvalues of a complex Hermitian or real symmetric matrix. Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,... |
| `torch.linalg.householder_product` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | householder_product(A, tau, *, out=None) -> Tensor Computes the first `n` columns of a product of Householder matrices. Let :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, and let :... |
| `torch.linalg.inv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.inv(A, *, out=None) -> Tensor Computes the inverse of a square matrix if it exists. Throws a `RuntimeError` if the matrix is not invertible. Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` ... |
| `torch.linalg.inv_ex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.inv_ex(A, *, check_errors=False, out=None) -> (Tensor, Tensor) Computes the inverse of a square matrix if it is invertible. Returns a namedtuple ``(inverse, info)``. ``inverse`` contains the... |
| `torch.linalg.ldl_factor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.ldl_factor(A, *, hermitian=False, out=None) -> (Tensor, Tensor) Computes a compact representation of the LDL factorization of a Hermitian or symmetric (possibly indefinite) matrix. When :att... |
| `torch.linalg.ldl_factor_ex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.ldl_factor_ex(A, *, hermitian=False, check_errors=False, out=None) -> (Tensor, Tensor, Tensor) This is a version of :func:`~ldl_factor` that does not perform error checks unless :attr:`check... |
| `torch.linalg.ldl_solve` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.ldl_solve(LD, pivots, B, *, hermitian=False, out=None) -> Tensor Computes the solution of a system of linear equations using the LDL factorization. :attr:`LD` and :attr:`pivots` are the comp... |
| `torch.linalg.lstsq` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | torch.linalg.lstsq(A, B, rcond=None, *, driver=None) -> (Tensor, Tensor, Tensor, Tensor) Computes a solution to the least squares problem of a system of linear equations. Letting :math:`\mathbb{K}`... |
| `torch.linalg.lu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | lu(A, *, pivot=True, out=None) -> (Tensor, Tensor, Tensor) Computes the LU decomposition with partial pivoting of a matrix. Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, t... |
| `torch.linalg.lu_factor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.lu_factor(A, *, bool pivot=True, out=None) -> (Tensor, Tensor) Computes a compact representation of the LU factorization with partial pivoting of a matrix. This function computes a compact r... |
| `torch.linalg.lu_factor_ex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.lu_factor_ex(A, *, pivot=True, check_errors=False, out=None) -> (Tensor, Tensor, Tensor) This is a version of :func:`~lu_factor` that does not perform error checks unless :attr:`check_errors... |
| `torch.linalg.lu_solve` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.lu_solve(LU, pivots, B, *, left=True, adjoint=False, out=None) -> Tensor Computes the solution of a square system of linear equations with a unique solution given an LU decomposition. Lettin... |
| `torch.linalg.matmul` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.matmul(input, other, *, out=None) -> Tensor Alias for :func:`torch.matmul` |
| `torch.linalg.matrix_exp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.matrix_exp(A) -> Tensor Computes the matrix exponential of a square matrix. Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, this function computes the **matrix expone... |
| `torch.linalg.matrix_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None, out=None) -> Tensor Computes a matrix norm. If :attr:`A` is complex valued, it computes the norm of :attr:`A`\ `.abs()` ... |
| `torch.linalg.matrix_power` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | matrix_power(A, n, *, out=None) -> Tensor Computes the `n`-th power of a square matrix for an integer `n`. Supports input of float, double, cfloat and cdouble dtypes. Also supports batches of matri... |
| `torch.linalg.matrix_rank` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None) -> Tensor Computes the numerical rank of a matrix. The matrix rank is computed as the number of singular values (or eigenva... |
| `torch.linalg.multi_dot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.multi_dot(tensors, *, out=None) Efficiently multiplies two or more matrices by reordering the multiplications so that the fewest arithmetic operations are performed. Supports inputs of float... |
| `torch.linalg.norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None) -> Tensor Computes a vector or matrix norm. Supports input of float, double, cfloat and cdouble dtypes. Whether this funct... |
| `torch.linalg.pinv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.pinv(A, *, atol=None, rtol=None, hermitian=False, out=None) -> Tensor Computes the pseudoinverse (Moore-Penrose inverse) of a matrix. The pseudoinverse may be `defined algebraically`_ but it... |
| `torch.linalg.qr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | qr(A, mode='reduced', *, out=None) -> (Tensor, Tensor) Computes the QR decomposition of a matrix. Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, the **full QR decomposition... |
| `torch.linalg.slogdet` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.slogdet(A, *, out=None) -> (Tensor, Tensor) Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix. For complex :attr:`A`, it returns the sign an... |
| `torch.linalg.solve` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.solve(A, B, *, left=True, out=None) -> Tensor Computes the solution of a square system of linear equations with a unique solution. Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`... |
| `torch.linalg.solve_ex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.solve_ex(A, B, *, left=True, check_errors=False, out=None) -> (Tensor, Tensor) A version of :func:`~solve` that does not perform error checks unless :attr:`check_errors`\ `= True`. It also r... |
| `torch.linalg.solve_triangular` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None) -> Tensor Computes the solution of a triangular system of linear equations with a unique solution. Letting :math:`\... |
| `torch.linalg.svd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.svd(A, full_matrices=True, *, driver=None, out=None) -> (Tensor, Tensor, Tensor) Computes the singular value decomposition (SVD) of a matrix. Letting :math:`\mathbb{K}` be :math:`\mathbb{R}`... |
| `torch.linalg.svdvals` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.svdvals(A, *, driver=None, out=None) -> Tensor Computes the singular values of a matrix. Supports input of float, double, cfloat and cdouble dtypes. Also supports batches of matrices, and if... |
| `torch.linalg.tensorinv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.tensorinv(A, ind=2, *, out=None) -> Tensor Computes the multiplicative inverse of :func:`torch.tensordot`. If `m` is the product of the first :attr:`ind` dimensions of :attr:`A` and `n` is t... |
| `torch.linalg.tensorsolve` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.tensorsolve(A, B, dims=None, *, out=None) -> Tensor Computes the solution `X` to the system `torch.tensordot(A, X) = B`. If `m` is the product of the first :attr:`B`\ `.ndim` dimensions of :... |
| `torch.linalg.vander` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | vander(x, N=None) -> Tensor Generates a Vandermonde matrix. Returns the Vandermonde matrix :math:`V` .. math:: V = \begin{pmatrix} 1 & x_1 & x_1^2 & \dots & x_1^{N-1}\\ 1 & x_2 & x_2^2 & \dots & x_... |
| `torch.linalg.vecdot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.vecdot(x, y, *, dim=-1, out=None) -> Tensor Computes the dot product of two batches of vectors along a dimension. In symbols, this function computes .. math:: \sum_{i=1}^n \overline{x_i}y_i.... |
| `torch.linalg.vector_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linalg.vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor Computes a vector norm. If :attr:`x` is complex valued, it computes the norm of :attr:`x`\ `.abs()` Supports... |
| | | | | | | | | |
| 🟦 MASKED_OPERATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.masked.MaskedTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `data, mask, requires_grad` | `Any` |  |
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
| 🟦 MONITORING | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.monitor.Aggregation` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | These are types of aggregations that can be used to accumulate stats. Members: VALUE : VALUE returns the last value to be added. MEAN : MEAN computes the arithmetic mean of all the added values. CO... |
| `torch.monitor.Event` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Event represents a specific typed event to be logged. This can represent high-level data points such as loss or accuracy per epoch or more low-level aggregations such as through the Stats provided ... |
| `torch.monitor.EventHandlerHandle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | EventHandlerHandle is a wrapper type returned by ``register_event_handler`` used to unregister the handler via ``unregister_event_handler``. This cannot be directly initialized. |
| `torch.monitor.Stat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Stat is used to compute summary statistics in a performant way over fixed intervals. Stat logs the statistics as an Event once every ``window_size`` duration. When the window closes the stats are l... |
| `torch.monitor.TensorboardEventHandler` | ❓ | ❓ | ❓ | ❓ | 🔴 | `writer` | `None` | TensorboardEventHandler is an event handler that will write known events to the provided SummaryWriter. This currently only supports ``torch.monitor.Stat`` events which are logged as scalars. Examp... |
| `torch.monitor.data_value_t` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | data_value_t is one of ``str``, ``float``, ``int``, ``bool``. |
| `torch.monitor.log_event` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | log_event(event: torch._C._monitor.Event) -> None log_event logs the specified event to all of the registered event handlers. It's up to the event handlers to log the event out to the corresponding... |
| `torch.monitor.register_event_handler` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | register_event_handler(callback: Callable[[torch._C._monitor.Event], None]) -> torch._C._monitor.EventHandlerHandle register_event_handler registers a callback to be called whenever an event is log... |
| `torch.monitor.unregister_event_handler` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | unregister_event_handler(handler: torch._C._monitor.EventHandlerHandle) -> None unregister_event_handler unregisters the ``EventHandlerHandle`` returned after calling ``register_event_handler``. Af... |
| | | | | | | | | |
| 🟦 MPS_OPERATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.mps.Event` | ❓ | ❓ | ❓ | ❓ | 🔴 | `enable_timing` | `None` | Wrapper around an MPS event. MPS events are synchronization markers that can be used to monitor the device's progress, to accurately measure timing, and to synchronize MPS streams. Args: enable_tim... |
| `torch.mps.Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
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
| `torch.mtia.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch.mtia.DeferredMtiaCallError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Common base class for all non-exit exceptions. |
| `torch.mtia.Event` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Event(device, *, enable_timing) -> Event Query and record Stream status to identify or control dependencies across Stream and measure timing. Arguments: device (:class:`torch.device`, optional): th... |
| `torch.mtia.Stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Stream(device, *, priority) -> Stream An in-order queue of executing the respective tasks asynchronously in first in first out (FIFO) order. It can control or synchronize the execution of other Str... |
| `torch.mtia.StreamContext` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `Any` | Context-manager that selects a given stream. All MTIA kernels queued within its context will be enqueued on a selected stream. Args: Stream (Stream): selected stream. This manager is a no-op if it'... |
| `torch.mtia.Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.mtia.classproperty` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `Any` |  |
| `torch.mtia.current_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the index of a currently selected device. |
| `torch.mtia.current_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.Stream'>` | Return the currently selected :class:`Stream` for a given device. Args: device (torch.device or int, optional): selected device. Returns the currently selected :class:`Stream` for the current devic... |
| `torch.mtia.default_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.Stream'>` | Return the default :class:`Stream` for a given device. Args: device (torch.device or int, optional): selected device. Returns the default :class:`Stream` for the current device, given by :func:`~to... |
| `torch.mtia.device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `Any` | Context-manager that changes the selected device. Args: device (torch.device or int): device index to select. It's a no-op if this argument is a negative integer or ``None``. |
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
| `torch.multiprocessing.AuthenticationError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Common base class for all non-exit exceptions. |
| `torch.multiprocessing.BufferTooShort` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Common base class for all non-exit exceptions. |
| `torch.multiprocessing.Process` | ❓ | ❓ | ❓ | ❓ | 🔴 | `group, target, name, ...` | `Any` | Process objects represent activity that is run in a separate process The class is analogous to `threading.Thread` |
| `torch.multiprocessing.ProcessContext` | ❓ | ❓ | ❓ | ❓ | 🔴 | `processes, error_files` | `Any` |  |
| `torch.multiprocessing.ProcessError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Common base class for all non-exit exceptions. |
| `torch.multiprocessing.ProcessExitedException` | ❓ | ❓ | ❓ | ❓ | 🔴 | `msg, error_index, error_pid, ...` | `Any` | Exception raised when a process failed due to signal or exited with a specific code. |
| `torch.multiprocessing.ProcessRaisedException` | ❓ | ❓ | ❓ | ❓ | 🔴 | `msg, error_index, error_pid` | `Any` | Exception raised when a process failed due to an exception raised by the code. |
| `torch.multiprocessing.SpawnContext` | ❓ | ❓ | ❓ | ❓ | 🔴 | `processes, error_files` | `Any` |  |
| `torch.multiprocessing.TimeoutError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Common base class for all non-exit exceptions. |
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
| `torch.nested.DType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.nested.Device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.nested.SymInt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `node` | `Any` | Like an int (including magic methods), but redirects all operations on the wrapped node. This is used in particular to symbolically record operations in the symbolic shape workflow. |
| `torch.nested.Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.nested.as_nested_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ts, dtype, device, ...` | `<class 'torch.Tensor'>` | Constructs a nested tensor preserving autograd history from a tensor or a list / tuple of tensors. If a nested tensor is passed, it will be returned directly unless the device / dtype / layout diff... |
| `torch.nested.masked_select` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, mask` | `<class 'torch.Tensor'>` | Constructs a nested tensor given a strided tensor input and a strided mask, the resulting jagged layout nested tensor will have values retain values where the mask is equal to True. The dimensional... |
| `torch.nested.narrow` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor, dim, start, ...` | `<class 'torch.Tensor'>` | Constructs a nested tensor (which might be a view) from :attr:`tensor`, a strided tensor. This follows similar semantics to torch.Tensor.narrow, where in the :attr:`dim`-th dimension the new nested... |
| `torch.nested.nested_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `tensor_list, dtype, layout, ...` | `<class 'torch.Tensor'>` | Constructs a nested tensor with no autograd history (also known as a "leaf tensor", see :ref:`Autograd mechanics <autograd-mechanics>`) from :attr:`tensor_list` a list of tensors. Args: tensor_list... |
| `torch.nested.nested_tensor_from_jagged` | ❓ | ❓ | ❓ | ❓ | 🔴 | `values, offsets, lengths, ...` | `<class 'torch.Tensor'>` | Constructs a jagged layout nested tensor from the given jagged components. The jagged layout consists of a required values buffer with the jagged dimension packed into a single dimension. The offse... |
| `torch.nested.to_padded_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | to_padded_tensor(input, padding, output_size=None, out=None) -> Tensor Returns a new (non-nested) Tensor by padding the :attr:`input` nested tensor. The leading entries will be filled with the nest... |
| | | | | | | | | |
| 🟦 NEURAL_NETWORK | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.nn.AdaptiveAvgPool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `output_size` | `None` | Applies a 1D adaptive average pooling over an input signal composed of several input planes. The output size is :math:`L_{out}`, for any input size. The number of output features is equal to the nu... |
| `torch.nn.AdaptiveAvgPool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `output_size` | `None` | Applies a 2D adaptive average pooling over an input signal composed of several input planes. The output is of size H x W, for any input size. The number of output features is equal to the number of... |
| `torch.nn.AdaptiveAvgPool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `output_size` | `None` | Applies a 3D adaptive average pooling over an input signal composed of several input planes. The output is of size D x H x W, for any input size. The number of output features is equal to the numbe... |
| `torch.nn.AdaptiveLogSoftmaxWithLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `in_features, n_classes, cutoffs, ...` | `None` | Efficient softmax approximation. As described in `Efficient softmax approximation for GPUs by Edouard Grave, Armand Joulin, Moustapha Cissé, David Grangier, and Hervé Jégou <https://arxiv.org/abs/1... |
| `torch.nn.AdaptiveMaxPool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `output_size, return_indices` | `None` | Applies a 1D adaptive max pooling over an input signal composed of several input planes. The output size is :math:`L_{out}`, for any input size. The number of output features is equal to the number... |
| `torch.nn.AdaptiveMaxPool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `output_size, return_indices` | `None` | Applies a 2D adaptive max pooling over an input signal composed of several input planes. The output is of size :math:`H_{out} \times W_{out}`, for any input size. The number of output features is e... |
| `torch.nn.AdaptiveMaxPool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `output_size, return_indices` | `None` | Applies a 3D adaptive max pooling over an input signal composed of several input planes. The output is of size :math:`D_{out} \times H_{out} \times W_{out}`, for any input size. The number of outpu... |
| `torch.nn.AlphaDropout` | ❓ | ❓ | ❓ | ❓ | 🔴 | `p, inplace` | `None` | Applies Alpha Dropout over the input. Alpha Dropout is a type of Dropout that maintains the self-normalizing property. For an input with zero mean and unit standard deviation, the output of Alpha D... |
| `torch.nn.AvgPool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kernel_size, stride, padding, ...` | `None` | Applies a 1D average pooling over an input signal composed of several input planes. In the simplest case, the output value of the layer with input size :math:`(N, C, L)`, output :math:`(N, C, L_{ou... |
| `torch.nn.AvgPool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kernel_size, stride, padding, ...` | `None` | Applies a 2D average pooling over an input signal composed of several input planes. In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`, output :math:`(N, C, H_... |
| `torch.nn.AvgPool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kernel_size, stride, padding, ...` | `None` | Applies a 3D average pooling over an input signal composed of several input planes. In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`, output :math:`(N, C,... |
| `torch.nn.BCELoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `weight, size_average, reduce, ...` | `None` | Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities: The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as: ..... |
| `torch.nn.BCEWithLogitsLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `weight, size_average, reduce, ...` | `None` | This loss combines a `Sigmoid` layer and the `BCELoss` in one single class. This version is more numerically stable than using a plain `Sigmoid` followed by a `BCELoss` as, by combining the operati... |
| `torch.nn.BatchNorm1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `num_features, eps, momentum, ...` | `None` | Applies Batch Normalization over a 2D or 3D input. Method described in the paper `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs... |
| `torch.nn.BatchNorm2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `num_features, eps, momentum, ...` | `None` | Applies Batch Normalization over a 4D input. 4D is a mini-batch of 2D inputs with additional channel dimension. Method described in the paper `Batch Normalization: Accelerating Deep Network Trainin... |
| `torch.nn.BatchNorm3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `num_features, eps, momentum, ...` | `None` | Applies Batch Normalization over a 5D input. 5D is a mini-batch of 3D inputs with additional channel dimension as described in the paper `Batch Normalization: Accelerating Deep Network Training by ... |
| `torch.nn.Bilinear` | ❓ | ❓ | ❓ | ❓ | 🔴 | `in1_features, in2_features, out_features, ...` | `None` | Applies a bilinear transformation to the incoming data: :math:`y = x_1^T A x_2 + b`. Args: in1_features: size of each first input sample, must be > 0 in2_features: size of each second input sample,... |
| `torch.nn.Buffer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `data, persistent` | `Any` | A kind of Tensor that should not be considered a model parameter. For example, BatchNorm's ``running_mean`` is not a parameter, but is part of the module's state. Buffers are :class:`~torch.Tensor`... |
| `torch.nn.CELU` | ❓ | ❓ | ❓ | ❓ | 🔴 | `alpha, inplace` | `None` | Applies the CELU function element-wise. .. math:: \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1)) More details can be found in the paper `Continuously Differentiable Exponential... |
| `torch.nn.CTCLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `blank, reduction, zero_infinity` | `Any` | The Connectionist Temporal Classification loss. Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the probability of possible alignments of inp... |
| `torch.nn.ChannelShuffle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `groups` | `None` | Divides and rearranges the channels in a tensor. This operation divides the channels in a tensor of shape :math:`(N, C, *)` into g groups as :math:`(N, \frac{C}{g}, g, *)` and shuffles them, while ... |
| `torch.nn.CircularPad1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding` | `None` | Pads the input tensor using circular padding of the input boundary. Tensor values at the beginning of the dimension are used to pad the end, and values at the end are used to pad the beginning. If ... |
| `torch.nn.CircularPad2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding` | `None` | Pads the input tensor using circular padding of the input boundary. Tensor values at the beginning of the dimension are used to pad the end, and values at the end are used to pad the beginning. If ... |
| `torch.nn.CircularPad3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding` | `None` | Pads the input tensor using circular padding of the input boundary. Tensor values at the beginning of the dimension are used to pad the end, and values at the end are used to pad the beginning. If ... |
| `torch.nn.ConstantPad1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding, value` | `Any` | Pads the input tensor boundaries with a constant value. For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`. Args: padding (int, tuple): the size of the padding. If is `int`, uses th... |
| `torch.nn.ConstantPad2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding, value` | `None` | Pads the input tensor boundaries with a constant value. For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`. Args: padding (int, tuple): the size of the padding. If is `int`, uses th... |
| `torch.nn.ConstantPad3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding, value` | `None` | Pads the input tensor boundaries with a constant value. For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`. Args: padding (int, tuple): the size of the padding. If is `int`, uses th... |
| `torch.nn.Container` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing them to be nested in a tree structure. You can assign the su... |
| `torch.nn.Conv1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `in_channels, out_channels, kernel_size, ...` | `None` | Applies a 1D convolution over an input signal composed of several input planes. In the simplest case, the output value of the layer with input size :math:`(N, C_{\text{in}}, L)` and output :math:`(... |
| `torch.nn.Conv2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `in_channels, out_channels, kernel_size, ...` | `None` | Applies a 2D convolution over an input signal composed of several input planes. In the simplest case, the output value of the layer with input size :math:`(N, C_{\text{in}}, H, W)` and output :math... |
| `torch.nn.Conv3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `in_channels, out_channels, kernel_size, ...` | `None` | Applies a 3D convolution over an input signal composed of several input planes. In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)` and output :math:`(N... |
| `torch.nn.ConvTranspose1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `in_channels, out_channels, kernel_size, ...` | `None` | Applies a 1D transposed convolution operator over an input image composed of several input planes. This module can be seen as the gradient of Conv1d with respect to its input. It is also known as a... |
| `torch.nn.ConvTranspose2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `in_channels, out_channels, kernel_size, ...` | `None` | Applies a 2D transposed convolution operator over an input image composed of several input planes. This module can be seen as the gradient of Conv2d with respect to its input. It is also known as a... |
| `torch.nn.ConvTranspose3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `in_channels, out_channels, kernel_size, ...` | `None` | Applies a 3D transposed convolution operator over an input image composed of several input planes. The transposed convolution operator multiplies each input value element-wise by a learnable kernel... |
| `torch.nn.CosineEmbeddingLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `margin, size_average, reduce, ...` | `None` | Creates a criterion that measures the loss given input tensors :math:`x_1`, :math:`x_2` and a `Tensor` label :math:`y` with values 1 or -1. Use (:math:`y=1`) to maximize the cosine similarity of tw... |
| `torch.nn.CosineSimilarity` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dim, eps` | `None` | Returns cosine similarity between :math:`x_1` and :math:`x_2`, computed along `dim`. .. math :: \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)... |
| `torch.nn.CrossEntropyLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `weight, size_average, ignore_index, ...` | `None` | This criterion computes the cross entropy loss between input logits and target. It is useful when training a classification problem with `C` classes. If provided, the optional argument :attr:`weigh... |
| `torch.nn.CrossMapLRN2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size, alpha, beta, ...` | `None` | Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing them to be nested in a tree structure. You can assign the su... |
| `torch.nn.DataParallel` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module, device_ids, output_device, ...` | `None` | Implements data parallelism at the module level. This container parallelizes the application of the given :attr:`module` by splitting the input across the specified devices by chunking in the batch... |
| `torch.nn.Dropout` | ❓ | ❓ | ❓ | ❓ | 🔴 | `p, inplace` | `None` | During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p`. The zeroed elements are chosen independently for each forward call and are sampled from a Berno... |
| `torch.nn.Dropout1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `p, inplace` | `None` | Randomly zero out entire channels. A channel is a 1D feature map, e.g., the :math:`j`-th channel of the :math:`i`-th sample in the batched input is a 1D tensor :math:`\text{input}[i, j]`. Each chan... |
| `torch.nn.Dropout2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `p, inplace` | `None` | Randomly zero out entire channels. A channel is a 2D feature map, e.g., the :math:`j`-th channel of the :math:`i`-th sample in the batched input is a 2D tensor :math:`\text{input}[i, j]`. Each chan... |
| `torch.nn.Dropout3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `p, inplace` | `None` | Randomly zero out entire channels. A channel is a 3D feature map, e.g., the :math:`j`-th channel of the :math:`i`-th sample in the batched input is a 3D tensor :math:`\text{input}[i, j]`. Each chan... |
| `torch.nn.ELU` | ❓ | ❓ | ❓ | ❓ | 🔴 | `alpha, inplace` | `None` | Applies the Exponential Linear Unit (ELU) function, element-wise. Method described in the paper: `Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs) <https://arxiv.org/abs/1... |
| `torch.nn.Embedding` | ❓ | ❓ | ❓ | ❓ | 🔴 | `num_embeddings, embedding_dim, padding_idx, ...` | `None` | A simple lookup table that stores embeddings of a fixed dictionary and size. This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of... |
| `torch.nn.EmbeddingBag` | ❓ | ❓ | ❓ | ❓ | 🔴 | `num_embeddings, embedding_dim, max_norm, ...` | `None` | Compute sums or means of 'bags' of embeddings, without instantiating the intermediate embeddings. For bags of constant length, no :attr:`per_sample_weights`, no indices equal to :attr:`padding_idx`... |
| `torch.nn.FeatureAlphaDropout` | ❓ | ❓ | ❓ | ❓ | 🔴 | `p, inplace` | `None` | Randomly masks out entire channels. A channel is a feature map, e.g. the :math:`j`-th channel of the :math:`i`-th sample in the batch input is a tensor :math:`\text{input}[i, j]` of the input tenso... |
| `torch.nn.Flatten` | ❓ | ❓ | ❓ | ❓ | 🔴 | `start_dim, end_dim` | `None` | Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`, see :meth:`torch.flatten` for details. Shape: - Input: :math:`(*, S_{\text{start}},..., S_{i}, ..., S_{\text... |
| `torch.nn.Fold` | ❓ | ❓ | ❓ | ❓ | 🔴 | `output_size, kernel_size, dilation, ...` | `None` | Combines an array of sliding local blocks into a large containing tensor. Consider a batched :attr:`input` tensor containing sliding local blocks, e.g., patches of images, of shape :math:`(N, C \ti... |
| `torch.nn.FractionalMaxPool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kernel_size, output_size, output_ratio, ...` | `None` | Applies a 2D fractional max pooling over an input signal composed of several input planes. Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham The max-p... |
| `torch.nn.FractionalMaxPool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kernel_size, output_size, output_ratio, ...` | `None` | Applies a 3D fractional max pooling over an input signal composed of several input planes. Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham The max-p... |
| `torch.nn.GELU` | ❓ | ❓ | ❓ | ❓ | 🔴 | `approximate` | `None` | Applies the Gaussian Error Linear Units function. .. math:: \text{GELU}(x) = x * \Phi(x) where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution. When the approximat... |
| `torch.nn.GLU` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dim` | `None` | Applies the gated linear unit function. :math:`{GLU}(a, b)= a \otimes \sigma(b)` where :math:`a` is the first half of the input matrices and :math:`b` is the second half. Args: dim (int): the dimen... |
| `torch.nn.GRU` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | __init__(input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0.0,bidirectional=False,device=None,dtype=None) Apply a multi-layer gated recurrent unit (GRU) RNN to an input seque... |
| `torch.nn.GRUCell` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input_size, hidden_size, bias, ...` | `None` | A gated recurrent unit (GRU) cell. .. math:: \begin{array}{ll} r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\ z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\ n = \tanh(W_{in} x + b_{in} ... |
| `torch.nn.GaussianNLLLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `full, eps, reduction` | `None` | Gaussian negative log likelihood loss. The targets are treated as samples from Gaussian distributions with expectations and variances predicted by the neural network. For a ``target`` tensor modell... |
| `torch.nn.GroupNorm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `num_groups, num_channels, eps, ...` | `None` | Applies Group Normalization over a mini-batch of inputs. This layer implements the operation as described in the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__ .. math:: y = \frac... |
| `torch.nn.Hardshrink` | ❓ | ❓ | ❓ | ❓ | 🔴 | `lambd` | `None` | Applies the Hard Shrinkage (Hardshrink) function element-wise. Hardshrink is defined as: .. math:: \text{HardShrink}(x) = \begin{cases} x, & \text{ if } x > \lambda \\ x, & \text{ if } x < -\lambda... |
| `torch.nn.Hardsigmoid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `inplace` | `None` | Applies the Hardsigmoid function element-wise. Hardsigmoid is defined as: .. math:: \text{Hardsigmoid}(x) = \begin{cases} 0 & \text{if~} x \le -3, \\ 1 & \text{if~} x \ge +3, \\ x / 6 + 1 / 2 & \te... |
| `torch.nn.Hardswish` | ❓ | ❓ | ❓ | ❓ | 🔴 | `inplace` | `None` | Applies the Hardswish function, element-wise. Method described in the paper: `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_. Hardswish is defined as: .. math:: \text{Hardswish}(x) ... |
| `torch.nn.Hardtanh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `min_val, max_val, inplace, ...` | `None` | Applies the HardTanh function element-wise. HardTanh is defined as: .. math:: \text{HardTanh}(x) = \begin{cases} \text{max\_val} & \text{ if } x > \text{ max\_val } \\ \text{min\_val} & \text{ if }... |
| `torch.nn.HingeEmbeddingLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `margin, size_average, reduce, ...` | `None` | Measures the loss given an input tensor :math:`x` and a labels tensor :math:`y` (containing 1 or -1). This is usually used for measuring whether two inputs are similar or dissimilar, e.g. using the... |
| `torch.nn.HuberLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `reduction, delta` | `None` | Creates a criterion that uses a squared term if the absolute element-wise error falls below delta and a delta-scaled L1 term otherwise. This loss combines advantages of both :class:`L1Loss` and :cl... |
| `torch.nn.Identity` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `None` | A placeholder identity operator that is argument-insensitive. Args: args: any argument (unused) kwargs: any keyword argument (unused) Shape: - Input: :math:`(*)`, where :math:`*` means any number o... |
| `torch.nn.InstanceNorm1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `num_features, eps, momentum, ...` | `None` | Applies Instance Normalization. This operation applies Instance Normalization over a 2D (unbatched) or 3D (batched) input as described in the paper `Instance Normalization: The Missing Ingredient f... |
| `torch.nn.InstanceNorm2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `num_features, eps, momentum, ...` | `None` | Applies Instance Normalization. This operation applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper `Instance Norma... |
| `torch.nn.InstanceNorm3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `num_features, eps, momentum, ...` | `None` | Applies Instance Normalization. This operation applies Instance Normalization over a 5D input (a mini-batch of 3D inputs with additional channel dimension) as described in the paper `Instance Norma... |
| `torch.nn.KLDivLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size_average, reduce, reduction, ...` | `None` | The Kullback-Leibler divergence loss. For tensors of the same shape :math:`y_{\text{pred}},\ y_{\text{true}}`, where :math:`y_{\text{pred}}` is the :attr:`input` and :math:`y_{\text{true}}` is the ... |
| `torch.nn.L1Loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size_average, reduce, reduction` | `None` | Creates a criterion that measures the mean absolute error (MAE) between each element in the input :math:`x` and target :math:`y`. The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss ... |
| `torch.nn.LPPool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `norm_type, kernel_size, stride, ...` | `None` | Applies a 1D power-average pooling over an input signal composed of several input planes. On each window, the function computed is: .. math:: f(X) = \sqrt[p]{\sum_{x \in X} x^{p}} - At p = :math:`\... |
| `torch.nn.LPPool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `norm_type, kernel_size, stride, ...` | `None` | Applies a 2D power-average pooling over an input signal composed of several input planes. On each window, the function computed is: .. math:: f(X) = \sqrt[p]{\sum_{x \in X} x^{p}} - At p = :math:`\... |
| `torch.nn.LPPool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `norm_type, kernel_size, stride, ...` | `None` | Applies a 3D power-average pooling over an input signal composed of several input planes. On each window, the function computed is: .. math:: f(X) = \sqrt[p]{\sum_{x \in X} x^{p}} - At p = :math:`\... |
| `torch.nn.LSTM` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | __init__(input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0.0,bidirectional=False,proj_size=0,device=None,dtype=None) Apply a multi-layer long short-term memory (LSTM) RNN to... |
| `torch.nn.LSTMCell` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input_size, hidden_size, bias, ...` | `None` | A long short-term memory (LSTM) cell. .. math:: \begin{array}{ll} i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\ f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\ g = \tanh(W_{ig} x + b_{i... |
| `torch.nn.LayerNorm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `normalized_shape, eps, elementwise_affine, ...` | `None` | Applies Layer Normalization over a mini-batch of inputs. This layer implements the operation as described in the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__ .. math:: y = \frac... |
| `torch.nn.LazyBatchNorm1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `eps, momentum, affine, ...` | `None` | A :class:`torch.nn.BatchNorm1d` module with lazy initialization. Lazy initialization based on the ``num_features`` argument of the :class:`BatchNorm1d` that is inferred from the ``input.size(1)``. ... |
| `torch.nn.LazyBatchNorm2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `eps, momentum, affine, ...` | `None` | A :class:`torch.nn.BatchNorm2d` module with lazy initialization. Lazy initialization is done for the ``num_features`` argument of the :class:`BatchNorm2d` that is inferred from the ``input.size(1)`... |
| `torch.nn.LazyBatchNorm3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `eps, momentum, affine, ...` | `None` | A :class:`torch.nn.BatchNorm3d` module with lazy initialization. Lazy initialization is done for the ``num_features`` argument of the :class:`BatchNorm3d` that is inferred from the ``input.size(1)`... |
| `torch.nn.LazyConv1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `out_channels, kernel_size, stride, ...` | `None` | A :class:`torch.nn.Conv1d` module with lazy initialization of the ``in_channels`` argument. The ``in_channels`` argument of the :class:`Conv1d` is inferred from the ``input.size(1)``. The attribute... |
| `torch.nn.LazyConv2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `out_channels, kernel_size, stride, ...` | `None` | A :class:`torch.nn.Conv2d` module with lazy initialization of the ``in_channels`` argument. The ``in_channels`` argument of the :class:`Conv2d` that is inferred from the ``input.size(1)``. The attr... |
| `torch.nn.LazyConv3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `out_channels, kernel_size, stride, ...` | `None` | A :class:`torch.nn.Conv3d` module with lazy initialization of the ``in_channels`` argument. The ``in_channels`` argument of the :class:`Conv3d` that is inferred from the ``input.size(1)``. The attr... |
| `torch.nn.LazyConvTranspose1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `out_channels, kernel_size, stride, ...` | `None` | A :class:`torch.nn.ConvTranspose1d` module with lazy initialization of the ``in_channels`` argument. The ``in_channels`` argument of the :class:`ConvTranspose1d` that is inferred from the ``input.s... |
| `torch.nn.LazyConvTranspose2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `out_channels, kernel_size, stride, ...` | `None` | A :class:`torch.nn.ConvTranspose2d` module with lazy initialization of the ``in_channels`` argument. The ``in_channels`` argument of the :class:`ConvTranspose2d` is inferred from the ``input.size(1... |
| `torch.nn.LazyConvTranspose3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `out_channels, kernel_size, stride, ...` | `None` | A :class:`torch.nn.ConvTranspose3d` module with lazy initialization of the ``in_channels`` argument. The ``in_channels`` argument of the :class:`ConvTranspose3d` is inferred from the ``input.size(1... |
| `torch.nn.LazyInstanceNorm1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `eps, momentum, affine, ...` | `None` | A :class:`torch.nn.InstanceNorm1d` module with lazy initialization of the ``num_features`` argument. The ``num_features`` argument of the :class:`InstanceNorm1d` is inferred from the ``input.size(1... |
| `torch.nn.LazyInstanceNorm2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `eps, momentum, affine, ...` | `None` | A :class:`torch.nn.InstanceNorm2d` module with lazy initialization of the ``num_features`` argument. The ``num_features`` argument of the :class:`InstanceNorm2d` is inferred from the ``input.size(1... |
| `torch.nn.LazyInstanceNorm3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `eps, momentum, affine, ...` | `None` | A :class:`torch.nn.InstanceNorm3d` module with lazy initialization of the ``num_features`` argument. The ``num_features`` argument of the :class:`InstanceNorm3d` is inferred from the ``input.size(1... |
| `torch.nn.LazyLinear` | ❓ | ❓ | ❓ | ❓ | 🔴 | `out_features, bias, device, ...` | `None` | A :class:`torch.nn.Linear` module where `in_features` is inferred. In this module, the `weight` and `bias` are of :class:`torch.nn.UninitializedParameter` class. They will be initialized after the ... |
| `torch.nn.LeakyReLU` | ❓ | ❓ | ❓ | ❓ | 🔴 | `negative_slope, inplace` | `None` | Applies the LeakyReLU function element-wise. .. math:: \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x) or .. math:: \text{LeakyReLU}(x) = \begin{cases} x, & \text{ if } x \ge... |
| `torch.nn.Linear` | ❓ | ❓ | ❓ | ❓ | 🔴 | `in_features, out_features, bias, ...` | `None` | Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`. This module supports :ref:`TensorFloat32<tf32_on_ampere>`. On certain ROCm devices, when using float16 inputs thi... |
| `torch.nn.LocalResponseNorm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size, alpha, beta, ...` | `None` | Applies local response normalization over an input signal. The input signal is composed of several input planes, where channels occupy the second dimension. Applies normalization across channels. .... |
| `torch.nn.LogSigmoid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `None` | Applies the Logsigmoid function element-wise. .. math:: \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right) Shape: - Input: :math:`(*)`, where :math:`*` means any number of dimensions... |
| `torch.nn.LogSoftmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dim` | `None` | Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional input Tensor. The LogSoftmax formulation can be simplified as: .. math:: \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i... |
| `torch.nn.MSELoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size_average, reduce, reduction` | `None` | Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input :math:`x` and target :math:`y`. The unreduced (i.e. with :attr:`reduction` set to ``'non... |
| `torch.nn.MarginRankingLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `margin, size_average, reduce, ...` | `None` | Creates a criterion that measures the loss given inputs :math:`x1`, :math:`x2`, two 1D mini-batch or 0D `Tensors`, and a label 1D mini-batch or 0D `Tensor` :math:`y` (containing 1 or -1). If :math:... |
| `torch.nn.MaxPool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kernel_size, stride, padding, ...` | `None` | Applies a 1D max pooling over an input signal composed of several input planes. In the simplest case, the output value of the layer with input size :math:`(N, C, L)` and output :math:`(N, C, L_{out... |
| `torch.nn.MaxPool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kernel_size, stride, padding, ...` | `None` | Applies a 2D max pooling over an input signal composed of several input planes. In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`, output :math:`(N, C, H_{out... |
| `torch.nn.MaxPool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kernel_size, stride, padding, ...` | `None` | Applies a 3D max pooling over an input signal composed of several input planes. In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`, output :math:`(N, C, D_{... |
| `torch.nn.MaxUnpool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kernel_size, stride, padding` | `None` | Computes a partial inverse of :class:`MaxPool1d`. :class:`MaxPool1d` is not fully invertible, since the non-maximal values are lost. :class:`MaxUnpool1d` takes in as input the output of :class:`Max... |
| `torch.nn.MaxUnpool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kernel_size, stride, padding` | `None` | Computes a partial inverse of :class:`MaxPool2d`. :class:`MaxPool2d` is not fully invertible, since the non-maximal values are lost. :class:`MaxUnpool2d` takes in as input the output of :class:`Max... |
| `torch.nn.MaxUnpool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kernel_size, stride, padding` | `None` | Computes a partial inverse of :class:`MaxPool3d`. :class:`MaxPool3d` is not fully invertible, since the non-maximal values are lost. :class:`MaxUnpool3d` takes in as input the output of :class:`Max... |
| `torch.nn.Mish` | ❓ | ❓ | ❓ | ❓ | 🔴 | `inplace` | `Any` | Applies the Mish function, element-wise. Mish: A Self Regularized Non-Monotonic Neural Activation Function. .. math:: \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x)) .. note:: See `Mish: A Sel... |
| `torch.nn.Module` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `None` | Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing them to be nested in a tree structure. You can assign the su... |
| `torch.nn.ModuleDict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `modules` | `None` | Holds submodules in a dictionary. :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary, but modules it contains are properly registered, and will be visible by all :class:`... |
| `torch.nn.ModuleList` | ❓ | ❓ | ❓ | ❓ | 🔴 | `modules` | `None` | Holds submodules in a list. :class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but modules it contains are properly registered, and will be visible by all :class:`~torch.nn.Mo... |
| `torch.nn.MultiLabelMarginLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size_average, reduce, reduction` | `None` | Creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`) and output :math:`y` (which is a 2D `Tensor` ... |
| `torch.nn.MultiLabelSoftMarginLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `weight, size_average, reduce, ...` | `None` | Creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy, between input :math:`x` and target :math:`y` of size :math:`(N, C)`. For each sample in the minibatch: .. ... |
| `torch.nn.MultiMarginLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `p, margin, weight, ...` | `None` | Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`) and output :math:`y` (which is a 1D tensor of targe... |
| `torch.nn.MultiheadAttention` | ❓ | ❓ | ❓ | ❓ | 🔴 | `embed_dim, num_heads, dropout, ...` | `None` | Allows the model to jointly attend to information from different representation subspaces. .. note:: See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>... |
| `torch.nn.NLLLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `weight, size_average, ignore_index, ...` | `None` | The negative log likelihood loss. It is useful to train a classification problem with `C` classes. If provided, the optional argument :attr:`weight` should be a 1D Tensor assigning weight to each o... |
| `torch.nn.NLLLoss2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | The negative log likelihood loss. It is useful to train a classification problem with `C` classes. If provided, the optional argument :attr:`weight` should be a 1D Tensor assigning weight to each o... |
| `torch.nn.PReLU` | ❓ | ❓ | ❓ | ❓ | 🔴 | `num_parameters, init, device, ...` | `None` | Applies the element-wise PReLU function. .. math:: \text{PReLU}(x) = \max(0,x) + a * \min(0,x) or .. math:: \text{PReLU}(x) = \begin{cases} x, & \text{ if } x \ge 0 \\ ax, & \text{ otherwise } \end... |
| `torch.nn.PairwiseDistance` | ❓ | ❓ | ❓ | ❓ | 🔴 | `p, eps, keepdim` | `None` | Computes the pairwise distance between input vectors, or between columns of input matrices. Distances are computed using ``p``-norm, with constant ``eps`` added to avoid division by zero if ``p`` i... |
| `torch.nn.Parameter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `data, requires_grad` | `Any` | A kind of Tensor that is to be considered a module parameter. Parameters are :class:`~torch.Tensor` subclasses, that have a very special property when used with :class:`Module` s - when they're ass... |
| `torch.nn.ParameterDict` | ❓ | ❓ | ❓ | ❓ | 🔴 | `parameters` | `None` | Holds parameters in a dictionary. ParameterDict can be indexed like a regular Python dictionary, but Parameters it contains are properly registered, and will be visible by all Module methods. Other... |
| `torch.nn.ParameterList` | ❓ | ❓ | ❓ | ❓ | 🔴 | `values` | `None` | Holds parameters in a list. :class:`~torch.nn.ParameterList` can be used like a regular Python list, but Tensors that are :class:`~torch.nn.Parameter` are properly registered, and will be visible b... |
| `torch.nn.PixelShuffle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `upscale_factor` | `None` | Rearrange elements in a tensor according to an upscaling factor. Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)` to a tensor of shape :math:`(*, C, H \times r, W \times r)`... |
| `torch.nn.PixelUnshuffle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `downscale_factor` | `None` | Reverse the PixelShuffle operation. Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements in a tensor of shape :math:`(*, C, H \times r, W \times r)` to a tensor of shape :... |
| `torch.nn.PoissonNLLLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `log_input, full, size_average, ...` | `None` | Negative log likelihood loss with Poisson distribution of target. The loss can be described as: .. math:: \text{target} \sim \mathrm{Poisson}(\text{input}) \text{loss}(\text{input}, \text{target}) ... |
| `torch.nn.RMSNorm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `normalized_shape, eps, elementwise_affine, ...` | `None` | Applies Root Mean Square Layer Normalization over a mini-batch of inputs. This layer implements the operation as described in the paper `Root Mean Square Layer Normalization <https://arxiv.org/pdf/... |
| `torch.nn.RNN` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | __init__(input_size,hidden_size,num_layers=1,nonlinearity='tanh',bias=True,batch_first=False,dropout=0.0,bidirectional=False,device=None,dtype=None) Apply a multi-layer Elman RNN with :math:`\tanh`... |
| `torch.nn.RNNBase` | ❓ | ❓ | ❓ | ❓ | 🔴 | `mode, input_size, hidden_size, ...` | `None` | Base class for RNN modules (RNN, LSTM, GRU). Implements aspects of RNNs shared by the RNN, LSTM, and GRU classes, such as module initialization and utility methods for parameter storage management.... |
| `torch.nn.RNNCell` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input_size, hidden_size, bias, ...` | `None` | An Elman RNN cell with tanh or ReLU non-linearity. .. math:: h' = \tanh(W_{ih} x + b_{ih} + W_{hh} h + b_{hh}) If :attr:`nonlinearity` is `'relu'`, then ReLU is used in place of tanh. Args: input_s... |
| `torch.nn.RNNCellBase` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input_size, hidden_size, bias, ...` | `None` | Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing them to be nested in a tree structure. You can assign the su... |
| `torch.nn.RReLU` | ❓ | ❓ | ❓ | ❓ | 🔴 | `lower, upper, inplace` | `Any` | Applies the randomized leaky rectified linear unit function, element-wise. Method described in the paper: `Empirical Evaluation of Rectified Activations in Convolutional Network <https://arxiv.org/... |
| `torch.nn.ReLU` | ❓ | ❓ | ❓ | ❓ | 🔴 | `inplace` | `Any` | Applies the rectified linear unit function element-wise. :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)` Args: inplace: can optionally do the operation in-place. Default: ``False`` Shape: - Input: :mat... |
| `torch.nn.ReLU6` | ❓ | ❓ | ❓ | ❓ | 🔴 | `inplace` | `Any` | Applies the ReLU6 function element-wise. .. math:: \text{ReLU6}(x) = \min(\max(0,x), 6) Args: inplace: can optionally do the operation in-place. Default: ``False`` Shape: - Input: :math:`(*)`, wher... |
| `torch.nn.ReflectionPad1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding` | `None` | Pads the input tensor using the reflection of the input boundary. For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`. Args: padding (int, tuple): the size of the padding. If is `int... |
| `torch.nn.ReflectionPad2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding` | `None` | Pads the input tensor using the reflection of the input boundary. For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`. Args: padding (int, tuple): the size of the padding. If is `int... |
| `torch.nn.ReflectionPad3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding` | `None` | Pads the input tensor using the reflection of the input boundary. For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`. Args: padding (int, tuple): the size of the padding. If is `int... |
| `torch.nn.ReplicationPad1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding` | `None` | Pads the input tensor using replication of the input boundary. For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`. Args: padding (int, tuple): the size of the padding. If is `int`, ... |
| `torch.nn.ReplicationPad2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding` | `None` | Pads the input tensor using replication of the input boundary. For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`. Args: padding (int, tuple): the size of the padding. If is `int`, ... |
| `torch.nn.ReplicationPad3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding` | `None` | Pads the input tensor using replication of the input boundary. For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`. Args: padding (int, tuple): the size of the padding. If is `int`, ... |
| `torch.nn.SELU` | ❓ | ❓ | ❓ | ❓ | 🔴 | `inplace` | `None` | Applies the SELU function element-wise. .. math:: \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1))) with :math:`\alpha = 1.6732632423543772848170429916717` and :math:`\t... |
| `torch.nn.Sequential` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args` | `Any` | A sequential container. Modules will be added to it in the order they are passed in the constructor. Alternatively, an ``OrderedDict`` of modules can be passed in. The ``forward()`` method of ``Seq... |
| `torch.nn.SiLU` | ❓ | ❓ | ❓ | ❓ | 🔴 | `inplace` | `Any` | Applies the Sigmoid Linear Unit (SiLU) function, element-wise. The SiLU function is also known as the swish function. .. math:: \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the... |
| `torch.nn.Sigmoid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `None` | Applies the Sigmoid function element-wise. .. math:: \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)} Shape: - Input: :math:`(*)`, where :math:`*` means any number of dimensions. - Output: :m... |
| `torch.nn.SmoothL1Loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size_average, reduce, reduction, ...` | `None` | Creates a criterion that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise. It is less sensitive to outliers than :class:`torch.nn.MSELoss` and in som... |
| `torch.nn.SoftMarginLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size_average, reduce, reduction` | `None` | Creates a criterion that optimizes a two-class classification logistic loss between input tensor :math:`x` and target tensor :math:`y` (containing 1 or -1). .. math:: \text{loss}(x, y) = \sum_i \fr... |
| `torch.nn.Softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dim` | `None` | Applies the Softmax function to an n-dimensional input Tensor. Rescales them so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1. Softmax is defined as: .. m... |
| `torch.nn.Softmax2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `None` | Applies SoftMax over features to each spatial location. When given an image of ``Channels x Height x Width``, it will apply `Softmax` to each location :math:`(Channels, h_i, w_j)` Shape: - Input: :... |
| `torch.nn.Softmin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dim` | `None` | Applies the Softmin function to an n-dimensional input Tensor. Rescales them so that the elements of the n-dimensional output Tensor lie in the range `[0, 1]` and sum to 1. Softmin is defined as: .... |
| `torch.nn.Softplus` | ❓ | ❓ | ❓ | ❓ | 🔴 | `beta, threshold` | `None` | Applies the Softplus function element-wise. .. math:: \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) SoftPlus is a smooth approximation to the ReLU function and can be used to con... |
| `torch.nn.Softshrink` | ❓ | ❓ | ❓ | ❓ | 🔴 | `lambd` | `None` | Applies the soft shrinkage function element-wise. .. math:: \text{SoftShrinkage}(x) = \begin{cases} x - \lambda, & \text{ if } x > \lambda \\ x + \lambda, & \text{ if } x < -\lambda \\ 0, & \text{ ... |
| `torch.nn.Softsign` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `None` | Applies the element-wise Softsign function. .. math:: \text{SoftSign}(x) = \frac{x}{ 1 + |x|} Shape: - Input: :math:`(*)`, where :math:`*` means any number of dimensions. - Output: :math:`(*)`, sam... |
| `torch.nn.SyncBatchNorm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `num_features, eps, momentum, ...` | `None` | Applies Batch Normalization over a N-Dimensional input. The N-D input is a mini-batch of [N-2]D inputs with additional channel dimension) as described in the paper `Batch Normalization: Acceleratin... |
| `torch.nn.Tanh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `None` | Applies the Hyperbolic Tangent (Tanh) function element-wise. Tanh is defined as: .. math:: \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)} Shape: - Input: :math:`(*)`, wh... |
| `torch.nn.Tanhshrink` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `None` | Applies the element-wise Tanhshrink function. .. math:: \text{Tanhshrink}(x) = x - \tanh(x) Shape: - Input: :math:`(*)`, where :math:`*` means any number of dimensions. - Output: :math:`(*)`, same ... |
| `torch.nn.Threshold` | ❓ | ❓ | ❓ | ❓ | 🔴 | `threshold, value, inplace` | `None` | Thresholds each element of the input Tensor. Threshold is defined as: .. math:: y = \begin{cases} x, &\text{ if } x > \text{threshold} \\ \text{value}, &\text{ otherwise } \end{cases} Args: thresho... |
| `torch.nn.Transformer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `d_model, nhead, num_encoder_layers, ...` | `None` | A transformer model. .. note:: See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_ for an in depth discussion of the performant building blocks PyTorc... |
| `torch.nn.TransformerDecoder` | ❓ | ❓ | ❓ | ❓ | 🔴 | `decoder_layer, num_layers, norm` | `None` | TransformerDecoder is a stack of N decoder layers. .. note:: See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_ for an in depth discussion of the per... |
| `torch.nn.TransformerDecoderLayer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `d_model, nhead, dim_feedforward, ...` | `None` | TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network. .. note:: See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`... |
| `torch.nn.TransformerEncoder` | ❓ | ❓ | ❓ | ❓ | 🔴 | `encoder_layer, num_layers, norm, ...` | `None` | TransformerEncoder is a stack of N encoder layers. .. note:: See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_ for an in depth discussion of the per... |
| `torch.nn.TransformerEncoderLayer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `d_model, nhead, dim_feedforward, ...` | `None` | TransformerEncoderLayer is made up of self-attn and feedforward network. .. note:: See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_ for an in depth... |
| `torch.nn.TripletMarginLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `margin, p, eps, ...` | `Any` | Creates a criterion that measures the triplet loss given an input tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`. This is used for measuring a relative ... |
| `torch.nn.TripletMarginWithDistanceLoss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `distance_function, margin, swap, ...` | `Any` | Creates a criterion that measures the triplet loss given input tensors :math:`a`, :math:`p`, and :math:`n` (representing anchor, positive, and negative examples, respectively), and a nonnegative, r... |
| `torch.nn.Unflatten` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dim, unflattened_size` | `None` | Unflattens a tensor dim expanding it to a desired shape. For use with :class:`~nn.Sequential`. * :attr:`dim` specifies the dimension of the input tensor to be unflattened, and it can be either `int... |
| `torch.nn.Unfold` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kernel_size, dilation, padding, ...` | `None` | Extracts sliding local blocks from a batched input tensor. Consider a batched :attr:`input` tensor of shape :math:`(N, C, *)`, where :math:`N` is the batch dimension, :math:`C` is the channel dimen... |
| `torch.nn.UninitializedBuffer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `requires_grad, device, dtype, ...` | `None` | A buffer that is not initialized. Uninitialized Buffer is a a special case of :class:`torch.Tensor` where the shape of the data is still unknown. Unlike a :class:`torch.Tensor`, uninitialized param... |
| `torch.nn.UninitializedParameter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `requires_grad, device, dtype` | `None` | A parameter that is not initialized. Uninitialized Parameters are a special case of :class:`torch.nn.Parameter` where the shape of the data is still unknown. Unlike a :class:`torch.nn.Parameter`, u... |
| `torch.nn.Upsample` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size, scale_factor, mode, ...` | `None` | Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data. The input data is assumed to be of the form `minibatch x channels x [optional depth] x [optional height] x width... |
| `torch.nn.UpsamplingBilinear2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size, scale_factor` | `None` | Applies a 2D bilinear upsampling to an input signal composed of several input channels. To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor` as it's constructor argume... |
| `torch.nn.UpsamplingNearest2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `size, scale_factor` | `None` | Applies a 2D nearest neighbor upsampling to an input signal composed of several input channels. To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor` as it's constructo... |
| `torch.nn.ZeroPad1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding` | `None` | Pads the input tensor boundaries with zero. For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`. Args: padding (int, tuple): the size of the padding. If is `int`, uses the same paddi... |
| `torch.nn.ZeroPad2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding` | `None` | Pads the input tensor boundaries with zero. For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`. Args: padding (int, tuple): the size of the padding. If is `int`, uses the same paddi... |
| `torch.nn.ZeroPad3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `padding` | `None` | Pads the input tensor boundaries with zero. For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`. Args: padding (int, tuple): the size of the padding. If is `int`, uses the same paddi... |
| `torch.nn.factory_kwargs` | ❓ | ❓ | ❓ | ❓ | 🔴 | `kwargs` | `Any` | Return a canonicalized dict of factory kwargs. Given kwargs, returns a canonicalized dict of factory kwargs that can be directly passed to factory functions like torch.empty, or errors if unrecogni... |
| | | | | | | | | |
| 🟦 NEURAL_NETWORK_FUNCTIONAL | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.nn.functional.DType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | int([x]) -> integer int(x, base=10) -> integer Convert a number or string to an integer, or return 0 if no arguments are given. If x is a number, return x.__int__(). For floating point numbers, thi... |
| `torch.nn.functional.Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.nn.functional.adaptive_avg_pool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | adaptive_avg_pool1d(input, output_size) -> Tensor Applies a 1D adaptive average pooling over an input signal composed of several input planes. See :class:`~torch.nn.AdaptiveAvgPool1d` for details a... |
| `torch.nn.functional.adaptive_avg_pool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, output_size` | `<class 'torch.Tensor'>` | Apply a 2D adaptive average pooling over an input signal composed of several input planes. See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape. Args: output_size: the target outpu... |
| `torch.nn.functional.adaptive_avg_pool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, output_size` | `<class 'torch.Tensor'>` | Apply a 3D adaptive average pooling over an input signal composed of several input planes. See :class:`~torch.nn.AdaptiveAvgPool3d` for details and output shape. Args: output_size: the target outpu... |
| `torch.nn.functional.adaptive_max_pool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | adaptive_max_pool1d(input, output_size, return_indices=False) Applies a 1D adaptive max pooling over an input signal composed of several input planes. See :class:`~torch.nn.AdaptiveMaxPool1d` for d... |
| `torch.nn.functional.adaptive_max_pool1d_with_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, output_size, return_indices` | `tuple[torch.Tensor, torch.Tensor]` | adaptive_max_pool1d(input, output_size, return_indices=False) Applies a 1D adaptive max pooling over an input signal composed of several input planes. See :class:`~torch.nn.AdaptiveMaxPool1d` for d... |
| `torch.nn.functional.adaptive_max_pool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | adaptive_max_pool2d(input, output_size, return_indices=False) Applies a 2D adaptive max pooling over an input signal composed of several input planes. See :class:`~torch.nn.AdaptiveMaxPool2d` for d... |
| `torch.nn.functional.adaptive_max_pool2d_with_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, output_size, return_indices` | `tuple[torch.Tensor, torch.Tensor]` | adaptive_max_pool2d(input, output_size, return_indices=False) Applies a 2D adaptive max pooling over an input signal composed of several input planes. See :class:`~torch.nn.AdaptiveMaxPool2d` for d... |
| `torch.nn.functional.adaptive_max_pool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | adaptive_max_pool3d(input, output_size, return_indices=False) Applies a 3D adaptive max pooling over an input signal composed of several input planes. See :class:`~torch.nn.AdaptiveMaxPool3d` for d... |
| `torch.nn.functional.adaptive_max_pool3d_with_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, output_size, return_indices` | `tuple[torch.Tensor, torch.Tensor]` | adaptive_max_pool3d(input, output_size, return_indices=False) Applies a 3D adaptive max pooling over an input signal composed of several input planes. See :class:`~torch.nn.AdaptiveMaxPool3d` for d... |
| `torch.nn.functional.affine_grid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `theta, size, align_corners` | `<class 'torch.Tensor'>` | Generate 2D or 3D flow field (sampling grid), given a batch of affine matrices :attr:`theta`. .. note:: This function is often used in conjunction with :func:`grid_sample` to build `Spatial Transfo... |
| `torch.nn.functional.alpha_dropout` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, p, training, ...` | `<class 'torch.Tensor'>` | Apply alpha dropout to the input. See :class:`~torch.nn.AlphaDropout` for details. |
| `torch.nn.functional.assert_int_or_pair` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arg, arg_name, message` | `None` |  |
| `torch.nn.functional.avg_pool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor Applies a 1D average pooling over an input signal composed of several input planes. See :cl... |
| `torch.nn.functional.avg_pool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor Applies 2D average-pooling operation in :math:`kH \times kW` regions... |
| `torch.nn.functional.avg_pool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor Applies 3D average-pooling operation in :math:`kT \times kH \times k... |
| `torch.nn.functional.batch_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, running_mean, running_var, ...` | `<class 'torch.Tensor'>` | Apply Batch Normalization for each channel across a batch of data. See :class:`~torch.nn.BatchNorm1d`, :class:`~torch.nn.BatchNorm2d`, :class:`~torch.nn.BatchNorm3d` for details. |
| `torch.nn.functional.bilinear` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bilinear(input1, input2, weight, bias=None) -> Tensor Applies a bilinear transformation to the incoming data: :math:`y = x_1^T A x_2 + b` Shape: - input1: :math:`(N, *, H_{in1})` where :math:`H_{in... |
| `torch.nn.functional.binary_cross_entropy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, weight, ...` | `<class 'torch.Tensor'>` | Compute Binary Cross Entropy between the target and input probabilities. See :class:`~torch.nn.BCELoss` for details. Args: input: Tensor of arbitrary shape as probabilities. target: Tensor of the s... |
| `torch.nn.functional.binary_cross_entropy_with_logits` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, weight, ...` | `<class 'torch.Tensor'>` | Compute Binary Cross Entropy between target and input logits. See :class:`~torch.nn.BCEWithLogitsLoss` for details. Args: input: Tensor of arbitrary shape as unnormalized scores (often referred to ... |
| `torch.nn.functional.boolean_dispatch` | ❓ | ❓ | ❓ | ❓ | 🔴 | `arg_name, arg_index, default, ...` | `Any` | Dispatches to either of 2 script functions based on a boolean argument. In TorchScript, the boolean argument must be constant so that the correct function to use can be determined at compile time. |
| `torch.nn.functional.celu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, alpha, inplace` | `<class 'torch.Tensor'>` | celu(input, alpha=1., inplace=False) -> Tensor Applies element-wise, :math:`\text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))`. See :class:`~torch.nn.CELU` for more details. |
| `torch.nn.functional.celu_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | celu_(input, alpha=1.) -> Tensor In-place version of :func:`~celu`. |
| `torch.nn.functional.channel_shuffle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | channel_shuffle(input, groups) -> Tensor Divide the channels in a tensor of shape :math:`(*, C , H, W)` into g groups and rearrange them as :math:`(*, C \frac g, g, H, W)`, while keeping the origin... |
| `torch.nn.functional.conv1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor Applies a 1D convolution over an input signal composed of several input planes. This operator supports :ref:`Te... |
| `torch.nn.functional.conv2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor Applies a 2D convolution over an input image composed of several input planes. This operator supports :ref:`Ten... |
| `torch.nn.functional.conv3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor Applies a 3D convolution over an input image composed of several input planes. This operator supports :ref:`Ten... |
| `torch.nn.functional.conv_tbc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Applies a 1-dimensional sequence convolution over an input sequence. Input and output dimensions are (Time, Batch, Channels) - hence TBC. Args: input: input tensor of shape :math:`(\text{sequence l... |
| `torch.nn.functional.conv_transpose1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor Applies a 1D transposed convolution operator over an input signal composed of sever... |
| `torch.nn.functional.conv_transpose2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor Applies a 2D transposed convolution operator over an input image composed of severa... |
| `torch.nn.functional.conv_transpose3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor Applies a 3D transposed convolution operator over an input image composed of severa... |
| `torch.nn.functional.cosine_embedding_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input1, input2, target, ...` | `<class 'torch.Tensor'>` | Compute the cosine embedding loss. See :class:`~torch.nn.CosineEmbeddingLoss` for details. Args: input1 (Tensor): Predicted values. input2 (Tensor): Predicted values. target (Tensor): Ground truth ... |
| `torch.nn.functional.cosine_similarity` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | cosine_similarity(x1, x2, dim=1, eps=1e-8) -> Tensor Returns cosine similarity between ``x1`` and ``x2``, computed along dim. ``x1`` and ``x2`` must be broadcastable to a common shape. ``dim`` refe... |
| `torch.nn.functional.cross_entropy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, weight, ...` | `<class 'torch.Tensor'>` | Compute the cross entropy loss between input logits and target. See :class:`~torch.nn.CrossEntropyLoss` for details. Args: input (Tensor) : Predicted unnormalized logits; see Shape section below fo... |
| `torch.nn.functional.ctc_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `log_probs, targets, input_lengths, ...` | `<class 'torch.Tensor'>` | Compute the Connectionist Temporal Classification loss. See :class:`~torch.nn.CTCLoss` for details. Note: In some circumstances when given tensors on a CUDA device and using CuDNN, this operator ma... |
| `torch.nn.functional.dropout` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, p, training, ...` | `<class 'torch.Tensor'>` | During training, randomly zeroes some elements of the input tensor with probability :attr:`p`. Uses samples from a Bernoulli distribution. See :class:`~torch.nn.Dropout` for details. Args: p: proba... |
| `torch.nn.functional.dropout1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, p, training, ...` | `<class 'torch.Tensor'>` | Randomly zero out entire channels (a channel is a 1D feature map). For example, the :math:`j`-th channel of the :math:`i`-th sample in the batched input is a 1D tensor :math:`\text{input}[i, j]` of... |
| `torch.nn.functional.dropout2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, p, training, ...` | `<class 'torch.Tensor'>` | Randomly zero out entire channels (a channel is a 2D feature map). For example, the :math:`j`-th channel of the :math:`i`-th sample in the batched input is a 2D tensor :math:`\text{input}[i, j]` of... |
| `torch.nn.functional.dropout3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, p, training, ...` | `<class 'torch.Tensor'>` | Randomly zero out entire channels (a channel is a 3D feature map). For example, the :math:`j`-th channel of the :math:`i`-th sample in the batched input is a 3D tensor :math:`\text{input}[i, j]` of... |
| `torch.nn.functional.elu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, alpha, inplace` | `<class 'torch.Tensor'>` | Apply the Exponential Linear Unit (ELU) function element-wise. See :class:`~torch.nn.ELU` for more details. |
| `torch.nn.functional.elu_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | elu_(input, alpha=1.) -> Tensor In-place version of :func:`~elu`. |
| `torch.nn.functional.embedding` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, weight, padding_idx, ...` | `<class 'torch.Tensor'>` | Generate a simple lookup table that looks up embeddings in a fixed dictionary and size. This module is often used to retrieve word embeddings using indices. The input to the module is a list of ind... |
| `torch.nn.functional.embedding_bag` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, weight, offsets, ...` | `<class 'torch.Tensor'>` | Compute sums, means or maxes of `bags` of embeddings. Calculation is done without instantiating the intermediate embeddings. See :class:`torch.nn.EmbeddingBag` for more details. Note: This operatio... |
| `torch.nn.functional.feature_alpha_dropout` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, p, training, ...` | `<class 'torch.Tensor'>` | Randomly masks out entire channels (a channel is a feature map). For example, the :math:`j`-th channel of the :math:`i`-th sample in the batch input is a tensor :math:`\text{input}[i, j]` of the in... |
| `torch.nn.functional.fold` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, output_size, kernel_size, ...` | `<class 'torch.Tensor'>` | Combine an array of sliding local blocks into a large containing tensor. .. warning:: Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported. See :class:`torch.nn.Fo... |
| `torch.nn.functional.fractional_max_pool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None) Applies 2D fractional max pooling over an input signal composed of several... |
| `torch.nn.functional.fractional_max_pool2d_with_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, kernel_size, output_size, ...` | `tuple[torch.Tensor, torch.Tensor]` | fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None) Applies 2D fractional max pooling over an input signal composed of several... |
| `torch.nn.functional.fractional_max_pool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None) Applies 3D fractional max pooling over an input signal composed of several... |
| `torch.nn.functional.fractional_max_pool3d_with_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, kernel_size, output_size, ...` | `tuple[torch.Tensor, torch.Tensor]` | fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None) Applies 3D fractional max pooling over an input signal composed of several... |
| `torch.nn.functional.gaussian_nll_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, var, ...` | `<class 'torch.Tensor'>` | Compute the Gaussian negative log likelihood loss. See :class:`~torch.nn.GaussianNLLLoss` for details. Args: input: Expectation of the Gaussian distribution. target: Sample from the Gaussian distri... |
| `torch.nn.functional.gelu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | gelu(input, approximate = 'none') -> Tensor When the approximate argument is 'none', it applies element-wise the function :math:`\text{GELU}(x) = x * \Phi(x)` where :math:`\Phi(x)` is the Cumulativ... |
| `torch.nn.functional.glu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim` | `<class 'torch.Tensor'>` | glu(input, dim=-1) -> Tensor The gated linear unit. Computes: .. math :: \text{GLU}(a, b) = a \otimes \sigma(b) where `input` is split in half along `dim` to form `a` and `b`, :math:`\sigma` is the... |
| `torch.nn.functional.grid_sample` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, grid, mode, ...` | `<class 'torch.Tensor'>` | Compute grid sample. Given an :attr:`input` and a flow-field :attr:`grid`, computes the ``output`` using :attr:`input` values and pixel locations from :attr:`grid`. Currently, only spatial (4-D) an... |
| `torch.nn.functional.group_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, num_groups, weight, ...` | `<class 'torch.Tensor'>` | Apply Group Normalization for last certain number of dimensions. See :class:`~torch.nn.GroupNorm` for details. |
| `torch.nn.functional.gumbel_softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `logits, tau, hard, ...` | `<class 'torch.Tensor'>` | Sample from the Gumbel-Softmax distribution (`Link 1`_ `Link 2`_) and optionally discretize. Args: logits: `[..., num_features]` unnormalized log probabilities tau: non-negative scalar temperature ... |
| `torch.nn.functional.handle_torch_function` | ❓ | ❓ | ❓ | ❓ | 🔴 | `public_api, relevant_args, args, ...` | `typing.Any` | Implement a function with checks for ``__torch_function__`` overrides. See torch::autograd::handle_torch_function for the equivalent of this function in the C++ implementation. Arguments --------- ... |
| `torch.nn.functional.hardshrink` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hardshrink(input, lambd=0.5) -> Tensor Applies the hard shrinkage function element-wise See :class:`~torch.nn.Hardshrink` for more details. |
| `torch.nn.functional.hardsigmoid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, inplace` | `<class 'torch.Tensor'>` | Apply the Hardsigmoid function element-wise. .. math:: \text{Hardsigmoid}(x) = \begin{cases} 0 & \text{if~} x \le -3, \\ 1 & \text{if~} x \ge +3, \\ x / 6 + 1 / 2 & \text{otherwise} \end{cases} Arg... |
| `torch.nn.functional.hardswish` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, inplace` | `<class 'torch.Tensor'>` | Apply hardswish function, element-wise. Follows implementation as described in the paper: `Searching for MobileNetV3`_. .. math:: \text{Hardswish}(x) = \begin{cases} 0 & \text{if~} x \le -3, \\ x &... |
| `torch.nn.functional.hardtanh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, min_val, max_val, ...` | `<class 'torch.Tensor'>` | hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Tensor Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more details. |
| `torch.nn.functional.hardtanh_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hardtanh_(input, min_val=-1., max_val=1.) -> Tensor In-place version of :func:`~hardtanh`. |
| `torch.nn.functional.has_torch_function` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Check for __torch_function__ implementations in the elements of an iterable or if a __torch_function__ mode is enabled. Considers exact ``Tensor`` s and ``Parameter`` s non-dispatchable. Use this t... |
| `torch.nn.functional.has_torch_function_unary` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Special case of `has_torch_function` for single inputs. Instead of: `has_torch_function((t,))` call: `has_torch_function_unary(t)` which skips unnecessary packing and unpacking work. |
| `torch.nn.functional.has_torch_function_variadic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Special case of `has_torch_function` that skips tuple creation. This uses the METH_FASTCALL protocol introduced in Python 3.7 Instead of: `has_torch_function((a, b))` call: `has_torch_function_vari... |
| `torch.nn.functional.hinge_embedding_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, margin, ...` | `<class 'torch.Tensor'>` | Compute the hinge embedding loss. See :class:`~torch.nn.HingeEmbeddingLoss` for details. Args: input (Tensor): Predicted values. target (Tensor): Ground truth values. margin (float, optional): Marg... |
| `torch.nn.functional.huber_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, reduction, ...` | `<class 'torch.Tensor'>` | Compute the Huber loss, with optional weighting. Function uses a squared term if the absolute element-wise error falls below delta and a delta-scaled L1 term otherwise. When delta equals 1, this lo... |
| `torch.nn.functional.instance_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, running_mean, running_var, ...` | `<class 'torch.Tensor'>` | Apply Instance Normalization independently for each channel in every data sample within a batch. See :class:`~torch.nn.InstanceNorm1d`, :class:`~torch.nn.InstanceNorm2d`, :class:`~torch.nn.Instance... |
| `torch.nn.functional.interpolate` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, size, scale_factor, ...` | `<class 'torch.Tensor'>` | Down/up samples the input. Tensor interpolated to either the given :attr:`size` or the given :attr:`scale_factor` The algorithm used for interpolation is determined by :attr:`mode`. Currently tempo... |
| `torch.nn.functional.kl_div` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, size_average, ...` | `<class 'torch.Tensor'>` | Compute the KL Divergence loss. Refer - The `Kullback-Leibler divergence Loss <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`__ See :class:`~torch.nn.KLDivLoss` for details. Args: inpu... |
| `torch.nn.functional.l1_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, size_average, ...` | `<class 'torch.Tensor'>` | Compute the L1 loss, with optional weighting. Function that takes the mean element-wise absolute value difference. See :class:`~torch.nn.L1Loss` for details. Args: input (Tensor): Predicted values.... |
| `torch.nn.functional.layer_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, normalized_shape, weight, ...` | `<class 'torch.Tensor'>` | Apply Layer Normalization for last certain number of dimensions. See :class:`~torch.nn.LayerNorm` for details. |
| `torch.nn.functional.leaky_relu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, negative_slope, inplace` | `<class 'torch.Tensor'>` | leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor Applies element-wise, :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)` See :class:`~torch.nn.LeakyReLU`... |
| `torch.nn.functional.leaky_relu_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | leaky_relu_(input, negative_slope=0.01) -> Tensor In-place version of :func:`~leaky_relu`. |
| `torch.nn.functional.linear` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | linear(input, weight, bias=None) -> Tensor Applies a linear transformation to the incoming data: :math:`y = xA^T + b`. This operation supports 2-D :attr:`weight` with :ref:`sparse layout<sparse-doc... |
| `torch.nn.functional.local_response_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, size, alpha, ...` | `<class 'torch.Tensor'>` | Apply local response normalization over an input signal. The input signal is composed of several input planes, where channels occupy the second dimension. Normalization is applied across channels. ... |
| `torch.nn.functional.log_softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, _stacklevel, ...` | `<class 'torch.Tensor'>` | Apply a softmax followed by a logarithm. While mathematically equivalent to log(softmax(x)), doing these two operations separately is slower and numerically unstable. This function uses an alternat... |
| `torch.nn.functional.logsigmoid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logsigmoid(input) -> Tensor Applies element-wise :math:`\text{LogSigmoid}(x_i) = \log \left(\frac{1}{1 + \exp(-x_i)}\right)` See :class:`~torch.nn.LogSigmoid` for more details. |
| `torch.nn.functional.lp_pool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, norm_type, kernel_size, ...` | `<class 'torch.Tensor'>` | Apply a 1D power-average pooling over an input signal composed of several input planes. If the sum of all inputs to the power of `p` is zero, the gradient is set to zero as well. See :class:`~torch... |
| `torch.nn.functional.lp_pool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, norm_type, kernel_size, ...` | `<class 'torch.Tensor'>` | Apply a 2D power-average pooling over an input signal composed of several input planes. If the sum of all inputs to the power of `p` is zero, the gradient is set to zero as well. See :class:`~torch... |
| `torch.nn.functional.lp_pool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, norm_type, kernel_size, ...` | `<class 'torch.Tensor'>` | Apply a 3D power-average pooling over an input signal composed of several input planes. If the sum of all inputs to the power of `p` is zero, the gradient is set to zero as well. See :class:`~torch... |
| `torch.nn.functional.margin_ranking_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input1, input2, target, ...` | `<class 'torch.Tensor'>` | Compute the margin ranking loss. See :class:`~torch.nn.MarginRankingLoss` for details. Args: input1 (Tensor): Predicted values. input2 (Tensor): Predicted values. target (Tensor): Ground truth valu... |
| `torch.nn.functional.max_pool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) Applies a 1D max pooling over an input signal composed of several input planes. .. note:: T... |
| `torch.nn.functional.max_pool1d_with_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, kernel_size, stride, ...` | `tuple[torch.Tensor, torch.Tensor]` | max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) Applies a 1D max pooling over an input signal composed of several input planes. .. note:: T... |
| `torch.nn.functional.max_pool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) Applies a 2D max pooling over an input signal composed of several input planes. .. note:: T... |
| `torch.nn.functional.max_pool2d_with_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, kernel_size, stride, ...` | `tuple[torch.Tensor, torch.Tensor]` | max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) Applies a 2D max pooling over an input signal composed of several input planes. .. note:: T... |
| `torch.nn.functional.max_pool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) Applies a 3D max pooling over an input signal composed of several input planes. .. note:: T... |
| `torch.nn.functional.max_pool3d_with_indices` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, kernel_size, stride, ...` | `tuple[torch.Tensor, torch.Tensor]` | max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) Applies a 3D max pooling over an input signal composed of several input planes. .. note:: T... |
| `torch.nn.functional.max_unpool1d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, indices, kernel_size, ...` | `<class 'torch.Tensor'>` | Compute a partial inverse of :class:`MaxPool1d`. See :class:`~torch.nn.MaxUnpool1d` for details. |
| `torch.nn.functional.max_unpool2d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, indices, kernel_size, ...` | `<class 'torch.Tensor'>` | Compute a partial inverse of :class:`MaxPool2d`. See :class:`~torch.nn.MaxUnpool2d` for details. |
| `torch.nn.functional.max_unpool3d` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, indices, kernel_size, ...` | `<class 'torch.Tensor'>` | Compute a partial inverse of :class:`MaxPool3d`. See :class:`~torch.nn.MaxUnpool3d` for details. |
| `torch.nn.functional.mish` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, inplace` | `<class 'torch.Tensor'>` | Apply the Mish function, element-wise. Mish: A Self Regularized Non-Monotonic Neural Activation Function. .. math:: \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x)) .. note:: See `Mish: A Self ... |
| `torch.nn.functional.mse_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, size_average, ...` | `<class 'torch.Tensor'>` | Compute the element-wise mean squared error, with optional weighting. See :class:`~torch.nn.MSELoss` for details. Args: input (Tensor): Predicted values. target (Tensor): Ground truth values. size_... |
| `torch.nn.functional.multi_head_attention_forward` | ❓ | ❓ | ❓ | ❓ | 🔴 | `query, key, value, ...` | `tuple[torch.Tensor, typing.Optional[torch.Tensor]]` | Forward method for MultiHeadAttention. .. note:: See `this tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_ for an in depth discussion of the performant buil... |
| `torch.nn.functional.multi_margin_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, p, ...` | `<class 'torch.Tensor'>` | Compute the multi margin loss, with optional weighting. See :class:`~torch.nn.MultiMarginLoss` for details. Args: input (Tensor): Predicted values. target (Tensor): Ground truth values. p (int, opt... |
| `torch.nn.functional.multilabel_margin_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, size_average, ...` | `<class 'torch.Tensor'>` | Compute the multilabel margin loss. See :class:`~torch.nn.MultiLabelMarginLoss` for details. Args: input (Tensor): Predicted values. target (Tensor): Ground truth values. size_average (bool, option... |
| `torch.nn.functional.multilabel_soft_margin_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, weight, ...` | `<class 'torch.Tensor'>` | Compute the multilabel soft margin loss. See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details. Args: input (Tensor): Predicted values. target (Tensor): Ground truth values. size_average (boo... |
| `torch.nn.functional.native_channel_shuffle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | native_channel_shuffle(input, groups) -> Tensor Native kernel level implementation of the `channel_shuffle`. This function might become private in future releases, use with caution. Divide the chan... |
| `torch.nn.functional.nll_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, weight, ...` | `<class 'torch.Tensor'>` | Compute the negative log likelihood loss. See :class:`~torch.nn.NLLLoss` for details. Args: input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)` in case of 2D Loss, or :math:... |
| `torch.nn.functional.normalize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, p, dim, ...` | `<class 'torch.Tensor'>` | Perform :math:`L_p` normalization of inputs over specified dimension. For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each :math:`n_{dim}` -element vector :math:`v` along... |
| `torch.nn.functional.one_hot` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | one_hot(tensor, num_classes=-1) -> LongTensor Takes LongTensor with index values of shape ``(*)`` and returns a tensor of shape ``(*, num_classes)`` that have zeros everywhere except where the inde... |
| `torch.nn.functional.pad` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, pad, mode, ...` | `<class 'torch.Tensor'>` | pad(input, pad, mode="constant", value=None) -> Tensor Pads tensor. Padding size: The padding size by which to pad some dimensions of :attr:`input` are described starting from the last dimension an... |
| `torch.nn.functional.pairwise_distance` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False) -> Tensor See :class:`torch.nn.PairwiseDistance` for details |
| `torch.nn.functional.pdist` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | pdist(input, p=2) -> Tensor Computes the p-norm distance between every pair of row vectors in the input. This is identical to the upper triangular portion, excluding the diagonal, of `torch.norm(in... |
| `torch.nn.functional.pixel_shuffle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | pixel_shuffle(input, upscale_factor) -> Tensor Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)` to a tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is the :... |
| `torch.nn.functional.pixel_unshuffle` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | pixel_unshuffle(input, downscale_factor) -> Tensor Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements in a tensor of shape :math:`(*, C, H \times r, W \times r)` to a te... |
| `torch.nn.functional.poisson_nll_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, log_input, ...` | `<class 'torch.Tensor'>` | Compute the Poisson negative log likelihood loss. See :class:`~torch.nn.PoissonNLLLoss` for details. Args: input: Expectation of underlying Poisson distribution. target: Random sample :math:`target... |
| `torch.nn.functional.prelu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | prelu(input, weight) -> Tensor Applies element-wise the function :math:`\text{PReLU}(x) = \max(0,x) + \text{weight} * \min(0,x)` where weight is a learnable parameter. .. note:: `weight` is expecte... |
| `torch.nn.functional.relu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, inplace` | `<class 'torch.Tensor'>` | relu(input, inplace=False) -> Tensor Applies the rectified linear unit function element-wise. See :class:`~torch.nn.ReLU` for more details. |
| `torch.nn.functional.relu6` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, inplace` | `<class 'torch.Tensor'>` | relu6(input, inplace=False) -> Tensor Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`. See :class:`~torch.nn.ReLU6` for more details. |
| `torch.nn.functional.relu_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | relu_(input) -> Tensor In-place version of :func:`~relu`. |
| `torch.nn.functional.rms_norm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, normalized_shape, weight, ...` | `<class 'torch.Tensor'>` | Apply Root Mean Square Layer Normalization. See :class:`~torch.nn.RMSNorm` for details. |
| `torch.nn.functional.rrelu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, lower, upper, ...` | `<class 'torch.Tensor'>` | rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Tensor Randomized leaky ReLU. See :class:`~torch.nn.RReLU` for more details. |
| `torch.nn.functional.rrelu_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | rrelu_(input, lower=1./8, upper=1./3, training=False) -> Tensor In-place version of :func:`~rrelu`. |
| `torch.nn.functional.scaled_dot_product_attention` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> Tensor: Computes scaled dot product attention on query, key and valu... |
| `torch.nn.functional.selu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, inplace` | `<class 'torch.Tensor'>` | selu(input, inplace=False) -> Tensor Applies element-wise, :math:`\text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))`, with :math:`\alpha=1.6732632423543772848170429916717` and ... |
| `torch.nn.functional.selu_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | selu_(input) -> Tensor In-place version of :func:`~selu`. |
| `torch.nn.functional.sigmoid` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input` | `Any` | sigmoid(input) -> Tensor Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}` See :class:`~torch.nn.Sigmoid` for more details. |
| `torch.nn.functional.silu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, inplace` | `<class 'torch.Tensor'>` | Apply the Sigmoid Linear Unit (SiLU) function, element-wise. The SiLU function is also known as the swish function. .. math:: \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the l... |
| `torch.nn.functional.smooth_l1_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, size_average, ...` | `<class 'torch.Tensor'>` | Compute the Smooth L1 loss. Function uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise. See :class:`~torch.nn.SmoothL1Loss` for details. Args: input (... |
| `torch.nn.functional.soft_margin_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, target, size_average, ...` | `<class 'torch.Tensor'>` | Compute the soft margin loss. See :class:`~torch.nn.SoftMarginLoss` for details. Args: input (Tensor): Predicted values. target (Tensor): Ground truth values. size_average (bool, optional): Depreca... |
| `torch.nn.functional.softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, _stacklevel, ...` | `<class 'torch.Tensor'>` | Apply a softmax function. Softmax is defined as: :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}` It is applied to all slices along dim, and will re-scale them so that the element... |
| `torch.nn.functional.softmin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, _stacklevel, ...` | `<class 'torch.Tensor'>` | Apply a softmin function. Note that :math:`\text{Softmin}(x) = \text{Softmax}(-x)`. See softmax definition for mathematical formula. See :class:`~torch.nn.Softmin` for more details. Args: input (Te... |
| `torch.nn.functional.softplus` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | softplus(input, beta=1, threshold=20) -> Tensor Applies element-wise, the function :math:`\text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))`. For numerical stability the implementati... |
| `torch.nn.functional.softshrink` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | softshrink(input, lambd=0.5) -> Tensor Applies the soft shrinkage function elementwise See :class:`~torch.nn.Softshrink` for more details. |
| `torch.nn.functional.softsign` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input` | `Any` | softsign(input) -> Tensor Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}` See :class:`~torch.nn.Softsign` for more details. |
| `torch.nn.functional.tanh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input` | `Any` | tanh(input) -> Tensor Applies element-wise, :math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}` See :class:`~torch.nn.Tanh` for more details. |
| `torch.nn.functional.tanhshrink` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input` | `Any` | tanhshrink(input) -> Tensor Applies element-wise, :math:`\text{Tanhshrink}(x) = x - \text{Tanh}(x)` See :class:`~torch.nn.Tanhshrink` for more details. |
| `torch.nn.functional.threshold` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, threshold, value, ...` | `<class 'torch.Tensor'>` | Apply a threshold to each element of the input Tensor. See :class:`~torch.nn.Threshold` for more details. |
| `torch.nn.functional.threshold_` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | threshold_(input, threshold, value) -> Tensor In-place version of :func:`~threshold`. |
| `torch.nn.functional.triplet_margin_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `anchor, positive, negative, ...` | `<class 'torch.Tensor'>` | Compute the triplet loss between given input tensors and a margin greater than 0. See :class:`~torch.nn.TripletMarginLoss` for details. |
| `torch.nn.functional.triplet_margin_with_distance_loss` | ❓ | ❓ | ❓ | ❓ | 🔴 | `anchor, positive, negative, ...` | `<class 'torch.Tensor'>` | Compute the triplet margin loss for input tensors using a custom distance function. See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details. |
| `torch.nn.functional.unfold` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, kernel_size, dilation, ...` | `<class 'torch.Tensor'>` | Extract sliding local blocks from a batched input tensor. .. warning:: Currently, only 4-D input tensors (batched image-like tensors) are supported. .. warning:: More than one element of the unfold... |
| `torch.nn.functional.upsample` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, size, scale_factor, ...` | `Any` | Upsample input. Provided tensor is upsampled to either the given :attr:`size` or the given :attr:`scale_factor` .. warning:: This function is deprecated in favor of :func:`torch.nn.functional.inter... |
| `torch.nn.functional.upsample_bilinear` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, size, scale_factor` | `Any` | Upsamples the input, using bilinear upsampling. .. warning:: This function is deprecated in favor of :func:`torch.nn.functional.interpolate`. This is equivalent with ``nn.functional.interpolate(...... |
| `torch.nn.functional.upsample_nearest` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, size, scale_factor` | `Any` | Upsamples the input, using nearest neighbours' pixel values. .. warning:: This function is deprecated in favor of :func:`torch.nn.functional.interpolate`. This is equivalent with ``nn.functional.in... |
| | | | | | | | | |
| 🟦 ONNX_EXPORT | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.onnx.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch.onnx.ExportOptions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Options for dynamo_export. .. deprecated:: 2.7 Please use ``torch.onnx.export(..., dynamo=True)`` instead. Attributes: dynamic_shapes: Shape information hint for input/output tensors. When ``None``... |
| `torch.onnx.JitScalarType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Scalar types defined in torch. Use ``JitScalarType`` to convert from torch and JIT scalar types to ONNX scalar types. Examples: >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX) >>> # xdoctest: +IG... |
| `torch.onnx.ONNXProgram` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, exported_program` | `Any` | A class to represent an ONNX program that is callable with torch tensors. Attributes: model: The ONNX model as an ONNX IR model object. exported_program: The exported program that produced the ONNX... |
| `torch.onnx.OnnxExporterError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Errors raised by the ONNX exporter. This is the base class for all exporter errors. |
| `torch.onnx.OperatorExportTypes` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: ONNX ONNX_ATEN ONNX_ATEN_FALLBACK ONNX_FALLTHROUGH |
| `torch.onnx.TensorProtoDataType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: UNDEFINED FLOAT UINT8 INT8 UINT16 INT16 INT32 INT64 STRING BOOL FLOAT16 DOUBLE UINT32 UINT64 COMPLEX64 COMPLEX128 BFLOAT16 FLOAT8E4M3FN FLOAT8E4M3FNUZ FLOAT8E5M2 FLOAT8E5M2FNUZ |
| `torch.onnx.TrainingMode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: EVAL PRESERVE TRAINING |
| `torch.onnx.deprecated` | ❓ | ❓ | ❓ | ❓ | 🔴 | `message, category, stacklevel` | `None` | Indicate that a class, function or overload is deprecated. When this decorator is applied to an object, the type checker will generate a diagnostic on usage of the deprecated object. Usage: @deprec... |
| `torch.onnx.dynamo_export` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, model_args, export_options, ...` | `ONNXProgram` | Export a torch.nn.Module to an ONNX graph. .. deprecated:: 2.7 Please use ``torch.onnx.export(..., dynamo=True)`` instead. Args: model: The PyTorch model to be exported to ONNX. model_args: Positio... |
| `torch.onnx.enable_fake_mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Enable fake mode for the duration of the context. Internally it instantiates a :class:`torch._subclasses.fake_tensor.FakeTensorMode` context manager that converts user input and model parameters in... |
| `torch.onnx.export` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, args, f, ...` | `ONNXProgram | None` | Exports a model into ONNX format. Setting ``dynamo=True`` enables the new ONNX export logic which is based on :class:`torch.export.ExportedProgram` and a more modern set of translation logic. This ... |
| `torch.onnx.is_in_onnx_export` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `bool` | Returns whether it is in the middle of ONNX export. |
| `torch.onnx.is_onnxrt_backend_supported` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` | Returns ``True`` if ONNX Runtime dependencies are installed and usable to support TorchDynamo backend integration; ``False`` otherwise. Example:: # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX) >>> i... |
| `torch.onnx.register_custom_op_symbolic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `symbolic_name, symbolic_fn, opset_version` | `Any` | Registers a symbolic function for a custom operator. When the user registers symbolic for custom/contrib ops, it is highly recommended to add shape inference for that operator via setType API, othe... |
| `torch.onnx.select_model_mode_for_export` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, mode` | `Any` | A context manager to temporarily set the training mode of ``model`` to ``mode``, resetting it when we exit the with-block. .. deprecated:: 2.7 Please set training mode before exporting the model. A... |
| `torch.onnx.unregister_custom_op_symbolic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `symbolic_name, opset_version` | `Any` | Unregisters ``symbolic_name``. See "Custom Operators" in the module documentation for an example usage. Args: symbolic_name (str): The name of the custom operator in "<domain>::<op>" format. opset_... |
| | | | | | | | | |
| 🟦 OPTIMIZATION | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.optim.ASGD` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, lambd, ...` | `Any` | Implements Averaged Stochastic Gradient Descent. It has been proposed in `Acceleration of stochastic approximation by averaging`_. Args: params (iterable): iterable of parameters or named_parameter... |
| `torch.optim.Adadelta` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, rho, ...` | `Any` | Implements Adadelta algorithm. .. math:: \begin{aligned} &\rule{110mm}{0.4pt} \\ &\textbf{input} : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)}, \: \rho \text... |
| `torch.optim.Adafactor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, beta2_decay, ...` | `Any` | Implements Adafactor algorithm. .. math:: \begin{aligned} &\rule{110mm}{0.4pt} \\ &\textbf{input} : \gamma \text{(lr)}, \: \tau \text{(}\beta_2\text{ decay)}, \: \theta_0 \text{(params)}, \: f(\the... |
| `torch.optim.Adagrad` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, lr_decay, ...` | `Any` | Implements Adagrad algorithm. .. math:: \begin{aligned} &\rule{110mm}{0.4pt} \\ &\textbf{input} : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)}, \: \lambda \te... |
| `torch.optim.Adam` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, betas, ...` | `Any` | Implements Adam algorithm. .. math:: \begin{aligned} &\rule{110mm}{0.4pt} \\ &\textbf{input} : \gamma \text{ (lr)}, \beta_1, \beta_2 \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (obje... |
| `torch.optim.AdamW` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, betas, ...` | `Any` | Implements AdamW algorithm, where weight decay does not accumulate in the momentum nor variance. .. math:: \begin{aligned} &\rule{110mm}{0.4pt} \\ &\textbf{input} : \gamma \text{(lr)}, \: \beta_1, ... |
| `torch.optim.Adamax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, betas, ...` | `Any` | Implements Adamax algorithm (a variant of Adam based on infinity norm). .. math:: \begin{aligned} &\rule{110mm}{0.4pt} \\ &\textbf{input} : \gamma \text{ (lr)}, \beta_1, \beta_2 \text{ (betas)},\th... |
| `torch.optim.LBFGS` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, max_iter, ...` | `Any` | Implements L-BFGS algorithm. Heavily inspired by `minFunc <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_. .. warning:: This optimizer doesn't support per-parameter options and parameter ... |
| `torch.optim.NAdam` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, betas, ...` | `Any` | Implements NAdam algorithm. .. math:: \begin{aligned} &\rule{110mm}{0.4pt} \\ &\textbf{input} : \gamma_t \text{ (lr)}, \: \beta_1,\beta_2 \text{ (betas)}, \: \theta_0 \text{ (params)}, \: f(\theta)... |
| `torch.optim.Optimizer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, defaults` | `None` | Base class for all optimizers. .. warning:: Parameters need to be specified as collections that have a deterministic ordering that is consistent between runs. Examples of objects that don't satisfy... |
| `torch.optim.RAdam` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, betas, ...` | `Any` | Implements RAdam algorithm. .. math:: \begin{aligned} &\rule{110mm}{0.4pt} \\ &\textbf{input} : \gamma \text{ (lr)}, \: \beta_1, \beta_2 \text{ (betas)}, \: \theta_0 \text{ (params)}, \:f(\theta) \... |
| `torch.optim.RMSprop` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, alpha, ...` | `Any` | Implements RMSprop algorithm. .. math:: \begin{aligned} &\rule{110mm}{0.4pt} \\ &\textbf{input} : \alpha \text{ (alpha)}, \: \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (... |
| `torch.optim.Rprop` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, etas, ...` | `Any` | Implements the resilient backpropagation algorithm. .. math:: \begin{aligned} &\rule{110mm}{0.4pt} \\ &\textbf{input} : \theta_0 \in \mathbf{R}^d \text{ (params)},f(\theta) \text{ (objective)}, \\ ... |
| `torch.optim.SGD` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, momentum, ...` | `Any` | Implements stochastic gradient descent (optionally with momentum). .. math:: \begin{aligned} &\rule{110mm}{0.4pt} \\ &\textbf{input} : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta... |
| `torch.optim.SparseAdam` | ❓ | ❓ | ❓ | ❓ | 🔴 | `params, lr, betas, ...` | `Any` | SparseAdam implements a masked version of the Adam algorithm suitable for sparse gradients. Currently, due to implementation constraints (explained below), SparseAdam is only intended for a narrow ... |
| | | | | | | | | |
| 🟦 CUSTOMIZATION | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.overrides.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch.overrides.BaseTorchFunctionMode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | A ``TorchFunctionMode`` allows you to override the meaning of all ``__torch_function__`` overrideable functions within a dynamic scope, without having to actually create a tensor subclass or manual... |
| `torch.overrides.Iterable` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.overrides.ParamSpec` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, bound, covariant, ...` | `Any` | Parameter specification. |
| `torch.overrides.TorchFunctionMode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | A ``TorchFunctionMode`` allows you to override the meaning of all ``__torch_function__`` overrideable functions within a dynamic scope, without having to actually create a tensor subclass or manual... |
| `torch.overrides.TypeVar` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, constraints, bound, ...` | `Any` | Type variable. Usage:: T = TypeVar('T') # Can be anything A = TypeVar('A', str, bytes) # Must be str or bytes Type variables exist primarily for the benefit of static type checkers. They serve as t... |
| `torch.overrides.enable_reentrant_dispatch` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.overrides.get_overridable_functions` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `dict[typing.Any, list[typing.Callable]]` | List functions that are overridable via __torch_function__ Returns ------- Dict[Any, List[Callable]] A dictionary that maps namespaces that contain overridable functions to functions in that namesp... |
| `torch.overrides.handle_torch_function` | ❓ | ❓ | ❓ | ❓ | 🔴 | `public_api, relevant_args, args, ...` | `typing.Any` | Implement a function with checks for ``__torch_function__`` overrides. See torch::autograd::handle_torch_function for the equivalent of this function in the C++ implementation. Arguments --------- ... |
| `torch.overrides.has_torch_function` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Check for __torch_function__ implementations in the elements of an iterable or if a __torch_function__ mode is enabled. Considers exact ``Tensor`` s and ``Parameter`` s non-dispatchable. Use this t... |
| `torch.overrides.has_torch_function_unary` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Special case of `has_torch_function` for single inputs. Instead of: `has_torch_function((t,))` call: `has_torch_function_unary(t)` which skips unnecessary packing and unpacking work. |
| `torch.overrides.has_torch_function_variadic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Special case of `has_torch_function` that skips tuple creation. This uses the METH_FASTCALL protocol introduced in Python 3.7 Instead of: `has_torch_function((a, b))` call: `has_torch_function_vari... |
| `torch.overrides.is_tensor_like` | ❓ | ❓ | ❓ | ❓ | 🔴 | `inp` | `Any` | Returns ``True`` if the passed-in input is a Tensor-like. Currently, this occurs whenever there's a ``__torch_function__`` attribute on the type of the input. Examples -------- A subclass of tensor... |
| `torch.overrides.is_tensor_method_or_property` | ❓ | ❓ | ❓ | ❓ | 🔴 | `func` | `<class 'bool'>` | Returns True if the function passed in is a handler for a method or property belonging to ``torch.Tensor``, as passed into ``__torch_function__``. .. note:: For properties, their ``__get__`` method... |
| `torch.overrides.resolve_name` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f` | `Any` | Get a human readable string name for a function passed to __torch_function__ Arguments --------- f : Callable Function to resolve the name of. Returns ------- str Name of the function; if eval'ed i... |
| `torch.overrides.wrap_torch_function` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dispatcher` | `Any` | Wraps a given function with ``__torch_function__`` -related functionality. Parameters ---------- dispatcher: Callable A callable that returns an iterable of Tensor-likes passed into the function. N... |
| `torch.overrides.wraps` | ❓ | ❓ | ❓ | ❓ | 🔴 | `wrapped, assigned, updated` | `Any` | Decorator factory to apply update_wrapper() to a wrapper function Returns a decorator that invokes update_wrapper() with the decorated function as the wrapper argument and the arguments to wraps() ... |
| | | | | | | | | |
| 🟦 PACKAGING | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.package.Directory` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, is_dir` | `Any` | A file structure representation. Organized as Directory nodes that have lists of their Directory children. Directories for a package are created by calling :meth:`PackageImporter.file_structure`. |
| `torch.package.EmptyMatchError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | This is an exception that is thrown when a mock or extern is marked as ``allow_empty=False``, and is not matched with any module during packaging. |
| `torch.package.GlobGroup` | ❓ | ❓ | ❓ | ❓ | 🔴 | `include, exclude, separator` | `Any` | A set of patterns that candidate strings will be matched against. A candidate is composed of a list of segments separated by ``separator``, e.g. "foo.bar.baz". A pattern contains one or more segmen... |
| `torch.package.Importer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Represents an environment to import modules from. By default, you can figure out what module an object belongs by checking __module__ and importing the result using __import__ or importlib.import_m... |
| `torch.package.ObjMismatchError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Raised when an importer found a different object with the same name as the user-provided one. |
| `torch.package.ObjNotFoundError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Raised when an importer cannot find an object by searching for its name. |
| `torch.package.OrderedImporter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args` | `Any` | A compound importer that takes a list of importers and tries them one at a time. The first importer in the list that returns a result "wins". |
| `torch.package.PackageExporter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `f, importer, debug` | `None` | Exporters allow you to write packages of code, pickled Python data, and arbitrary binary and text resources into a self-contained package. Imports can load this code in a hermetic way, such that co... |
| `torch.package.PackageImporter` | ❓ | ❓ | ❓ | ❓ | 🔴 | `file_or_buffer, module_allowed` | `Any` | Importers allow you to load code written to packages by :class:`PackageExporter`. Code is loaded in a hermetic way, using files from the package rather than the normal python import system. This al... |
| `torch.package.PackagingError` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dependency_graph, debug` | `Any` | This exception is raised when there is an issue with exporting a package. ``PackageExporter`` will attempt to gather up all the errors and present them to you at once. |
| `torch.package.is_from_package` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj` | `<class 'bool'>` | Return whether an object was loaded from a package. Note: packaged objects from externed modules will return ``False``. |
| | | | | | | | | |
| 🟦 PROFILING | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.profiler.DeviceType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: CPU CUDA MKLDNN OPENGL OPENCL IDEEP HIP FPGA MAIA XLA Vulkan Metal XPU MPS MTIA Meta HPU VE Lazy IPU PrivateUse1 |
| `torch.profiler.ExecutionTraceObserver` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Execution Trace Observer Each process can have a single ExecutionTraceObserver instance. The observer can be added to record function callbacks via calling register_callback() explicitly. Without c... |
| `torch.profiler.KinetoStepTracker` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Provides an abstraction for incrementing the step count globally. Previously, we only had one place to mark that a step() has occurred in the program via pytorch profiler step(). We will now add st... |
| `torch.profiler.ProfilerAction` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Profiler actions that can be taken at the specified intervals |
| `torch.profiler.ProfilerActivity` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: CPU XPU MTIA CUDA HPU PrivateUse1 |
| `torch.profiler.RecordScope` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: FUNCTION BACKWARD_FUNCTION TORCHSCRIPT_FUNCTION KERNEL_FUNCTION_DTYPE CUSTOM_CLASS BUILD_FEATURE LITE_INTERPRETER USER_SCOPE STATIC_RUNTIME_OP STATIC_RUNTIME_MODEL |
| `torch.profiler.is_fbcode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'bool'>` |  |
| `torch.profiler.kineto_available` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | kineto_available() -> bool |
| `torch.profiler.profile` | ❓ | ❓ | ❓ | ❓ | ⚠️ | `activities, schedule, on_trace_ready, ...` | `Any` | Profiler context manager. Args: activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values: ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerA... |
| `torch.profiler.record_function` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, args` | `Any` | Context manager/function decorator that adds a label to a code block/function when running autograd profiler. Label will only appear if CPU activity tracing is enabled. It is useful when tracing th... |
| `torch.profiler.register_optimizer_step_post_hook` | ❓ | ❓ | ❓ | ❓ | 🔴 | `hook` | `<class 'torch.utils.hooks.RemovableHandle'>` | Register a post hook common to all optimizers. The hook should have the following signature:: hook(optimizer, args, kwargs) -> None Args: hook (Callable): A user defined hook which is registered on... |
| `torch.profiler.schedule` | ❓ | ❓ | ❓ | ❓ | 🔴 | `wait, warmup, active, ...` | `typing.Callable` | Returns a callable that can be used as profiler ``schedule`` argument. The profiler will skip the first ``skip_first`` steps, then wait for ``wait`` steps, then do the warmup for the next ``warmup`... |
| `torch.profiler.supported_activities` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Returns a set of supported profiler tracing activities. Note: profiler uses CUPTI library to trace on-device CUDA kernels. In case when CUDA is enabled but CUPTI is not available, passing ``Profile... |
| `torch.profiler.tensorboard_trace_handler` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dir_name, worker_name, use_gzip` | `Any` | Outputs tracing files to directory of ``dir_name``, then that directory can be directly delivered to tensorboard as logdir. ``worker_name`` should be unique for each worker in distributed scenario,... |
| | | | | | | | | |
| 🟦 QUANTIZATION | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.quantization.ABC` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.quantization.DeQuantStub` | ❓ | ❓ | ❓ | ❓ | 🔴 | `qconfig` | `Any` | Dequantize stub module, before calibration, this is same as identity, this will be swapped as `nnq.DeQuantize` in `convert`. Args: qconfig: quantization configuration for the tensor, if qconfig is ... |
| `torch.quantization.FakeQuantize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `observer, quant_min, quant_max, ...` | `Any` | Simulate the quantize and dequantize operations in training time. The output of this module is given by:: x_out = ( clamp(round(x/scale + zero_point), quant_min, quant_max) - zero_point ) * scale *... |
| `torch.quantization.FakeQuantizeBase` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `None` | Base fake quantize module. Base fake quantize module Any fake quantize implementation should derive from this class. Concrete fake quantize module should follow the same API. In forward, they will ... |
| `torch.quantization.FixedQParamsFakeQuantize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `observer` | `Any` | Simulate quantize and dequantize in training time. Simulate quantize and dequantize with fixed quantization parameters in training time. Only per tensor quantization is supported. |
| `torch.quantization.FusedMovingAvgObsFakeQuantize` | ❓ | ❓ | ❓ | ❓ | 🔴 | `observer, quant_min, quant_max, ...` | `None` | Define a fused module to observe the tensor. Fused module that is used to observe the input tensor (compute min/max), compute scale/zero_point and fake_quantize the tensor. This module uses calcula... |
| `torch.quantization.HistogramObserver` | ❓ | ❓ | ❓ | ❓ | 🔴 | `bins, dtype, qscheme, ...` | `None` | The module records the running histogram of tensor values along with min/max values. ``calculate_qparams`` will calculate scale and zero_point. Args: bins: Number of bins to use for the histogram d... |
| `torch.quantization.MinMaxObserver` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype, qscheme, reduce_range, ...` | `None` | Observer module for computing the quantization parameters based on the running min and max values. This observer uses the tensor min/max statistics to compute the quantization parameters. The modul... |
| `torch.quantization.MovingAverageMinMaxObserver` | ❓ | ❓ | ❓ | ❓ | 🔴 | `averaging_constant, dtype, qscheme, ...` | `None` | Observer module for computing the quantization parameters based on the moving average of the min and max values. This observer computes the quantization parameters based on the moving averages of m... |
| `torch.quantization.MovingAveragePerChannelMinMaxObserver` | ❓ | ❓ | ❓ | ❓ | 🔴 | `averaging_constant, ch_axis, dtype, ...` | `None` | Observer module for computing the quantization parameters based on the running per channel min and max values. This observer uses the tensor min/max statistics to compute the per channel quantizati... |
| `torch.quantization.NoopObserver` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype, custom_op_name` | `None` | Observer that doesn't do anything and just passes its configuration to the quantized module's ``.from_float()``. Primarily used for quantization to float16 which doesn't require determining ranges.... |
| `torch.quantization.ObserverBase` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype, is_dynamic` | `Any` | Base observer Module. Any observer implementation should derive from this class. Concrete observers should follow the same API. In forward, they will update the statistics of the observed Tensor. A... |
| `torch.quantization.PerChannelMinMaxObserver` | ❓ | ❓ | ❓ | ❓ | 🔴 | `ch_axis, dtype, qscheme, ...` | `None` | Observer module for computing the quantization parameters based on the running per channel min and max values. This observer uses the tensor min/max statistics to compute the per channel quantizati... |
| `torch.quantization.PlaceholderObserver` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype, custom_op_name, compute_dtype, ...` | `None` | Observer that doesn't do anything and just passes its configuration to the quantized module's ``.from_float()``. Can be used for quantization to float16 which doesn't require determining ranges. Ar... |
| `torch.quantization.QConfig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `activation, weight` | `Any` | Describes how to quantize a layer or a part of the network by providing settings (observer classes) for activations and weights respectively. Note that QConfig needs to contain observer **classes**... |
| `torch.quantization.QConfigDynamic` | ❓ | ❓ | ❓ | ❓ | 🔴 | `activation, weight` | `Any` | Describes how to dynamically quantize a layer or a part of the network by providing settings (observer classes) for weights. It's like QConfig, but for dynamic quantization. Note that QConfigDynami... |
| `torch.quantization.QuantStub` | ❓ | ❓ | ❓ | ❓ | 🔴 | `qconfig` | `Any` | Quantize stub module, before calibration, this is same as an observer, it will be swapped as `nnq.Quantize` in `convert`. Args: qconfig: quantization configuration for the tensor, if qconfig is not... |
| `torch.quantization.QuantType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `value, names, module, ...` | `Any` | Enum where members are also (and must be) ints |
| `torch.quantization.QuantWrapper` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module` | `Any` | A wrapper class that wraps the input module, adds QuantStub and DeQuantStub and surround the call to module with call to quant and dequant modules. This is used by the `quantization` utility functi... |
| `torch.quantization.RecordingObserver` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `Any` | The module is mainly for debug and records the tensor values during runtime. Args: dtype: Quantized data type qscheme: Quantization scheme to be used reduce_range: Reduces the range of the quantize... |
| `torch.quantization.add_quant_dequant` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module` | `Any` | Wrap the leaf child module in QuantWrapper if it has a valid qconfig Note that this function will modify the children of module inplace and it can return a new module which wraps the input module a... |
| `torch.quantization.convert` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module, mapping, inplace, ...` | `Any` | Converts submodules in input module to a different module according to `mapping` by calling `from_float` method on the target module class. And remove qconfig at the end if remove_qconfig is set to... |
| `torch.quantization.convert_dynamic_jit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, inplace, debug, ...` | `Any` |  |
| `torch.quantization.convert_jit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, inplace, debug, ...` | `Any` |  |
| `torch.quantization.default_debug_observer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype` | `Any` | The module is mainly for debug and records the tensor values during runtime. Args: dtype: Quantized data type qscheme: Quantization scheme to be used reduce_range: Reduces the range of the quantize... |
| `torch.quantization.default_eval_fn` | ❓ | ❓ | ❓ | ❓ | 🔴 | `model, calib_data` | `Any` | Default evaluation function takes a torch.utils.data.Dataset or a list of input Tensors and run the model on the dataset |
| `torch.quantization.default_placeholder_observer` | ❓ | ❓ | ❓ | ❓ | 🔴 | `dtype, custom_op_name, compute_dtype, ...` | `None` | Observer that doesn't do anything and just passes its configuration to the quantized module's ``.from_float()``. Can be used for quantization to float16 which doesn't require determining ranges. Ar... |
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
| 🟦 RANDOM_GENERATION | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.random.Generator` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.random.fork_rng` | ❓ | ❓ | ❓ | ❓ | 🔴 | `devices, enabled, _caller, ...` | `<class 'collections.abc.Generator'>` | Forks the RNG, so that when you return, the RNG is reset to the state that it was previously in. Args: devices (iterable of Device IDs): devices for which to fork the RNG. CPU RNG state is always f... |
| `torch.random.get_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'torch.Tensor'>` | Returns the random number generator state as a `torch.ByteTensor`. .. note:: The returned state is for the default generator on CPU only. See also: :func:`torch.random.fork_rng`. |
| `torch.random.initial_seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Returns the initial seed for generating random numbers as a Python `long`. .. note:: The returned seed is for the default generator on CPU only. |
| `torch.random.manual_seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `seed` | `<class 'torch._C.Generator'>` | Sets the seed for generating random numbers on all devices. Returns a `torch.Generator` object. Args: seed (int): The desired seed. Value must be within the inclusive range `[-0x8000_0000_0000_0000... |
| `torch.random.seed` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Sets the seed for generating random numbers to a non-deterministic random number on all devices. Returns a 64 bit number used to seed the RNG. |
| `torch.random.set_rng_state` | ❓ | ❓ | ❓ | ❓ | 🔴 | `new_state` | `None` | Sets the random number generator state. .. note:: This function only works for CPU. For CUDA, please use :func:`torch.manual_seed`, which works for both CPU and CUDA. Args: new_state (torch.ByteTen... |
| | | | | | | | | |
| 🟦 RETURN_TYPES | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.return_types.SequenceKey` | ❓ | ❓ | ❓ | ❓ | 🔴 | `idx` | `None` | SequenceKey(idx: int) |
| `torch.return_types.aminmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.aminmax_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.cummax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.cummax_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.cummin` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.cummin_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.frexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.frexp_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.geqrf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.geqrf_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.histogram` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.histogram_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.histogramdd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.kthvalue` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.kthvalue_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_cholesky_ex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_cholesky_ex_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_eig` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_eig_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_eigh` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_eigh_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_inv_ex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_inv_ex_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_ldl_factor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_ldl_factor_ex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_ldl_factor_ex_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_ldl_factor_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_lstsq` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_lstsq_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_lu` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_lu_factor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_lu_factor_ex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_lu_factor_ex_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_lu_factor_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_lu_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_qr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_qr_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_slogdet` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_slogdet_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_solve_ex` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_solve_ex_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_svd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.linalg_svd_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.lu_unpack` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.lu_unpack_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.max` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.max_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.median` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.median_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.min` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.min_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.mode` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.mode_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.nanmedian` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.nanmedian_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.pytree_register_structseq` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cls` | `Any` |  |
| `torch.return_types.qr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.qr_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.register_pytree_node` | ❓ | ❓ | ❓ | ❓ | 🔴 | `cls, flatten_fn, unflatten_fn, ...` | `None` | Register a container-like type as pytree node. Note: :func:`register_dataclass` is a simpler way of registering a container-like type as a pytree node. Args: cls: the type to register flatten_fn: A... |
| `torch.return_types.slogdet` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.slogdet_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.sort` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.sort_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.svd` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.svd_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.topk` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.topk_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.triangular_solve` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.return_types.triangular_solve_out` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| | | | | | | | | |
| 🟦 SPARSE_OPERATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.sparse.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch.sparse.BFloat16Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sparse.ByteTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sparse.CharTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sparse.DType` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | int([x]) -> integer int(x, base=10) -> integer Convert a number or string to an integer, or return 0 if no arguments are given. If x is a number, return x.__int__(). For floating point numbers, thi... |
| `torch.sparse.DoubleTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sparse.FloatTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sparse.HalfTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sparse.IntTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sparse.LongTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sparse.ShortTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sparse.SparseSemiStructuredTensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, packed, meta, ...` | `Any` | This class implementes semi-structured sparsity as a Tensor subclass. Semi-structured sparsity describes a sparsity pattern where n in every 2n elements are sparse, depending on the datatype. It is... |
| `torch.sparse.SparseSemiStructuredTensorCUSPARSELT` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, packed, meta, ...` | `Any` | The cuSPARSELt backend expects the specified elements and the metadata to be stored in a single tensor: packed = [ specified elements of original tensor | metadata ] For an original tensor of size ... |
| `torch.sparse.SparseSemiStructuredTensorCUTLASS` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, packed, meta, ...` | `Any` | This class implements semi-structured sparsity for the CUTLASS backend. In this implementation, the specified elements and metadata are stored seprately, in packed and meta respectively. When _FORC... |
| `torch.sparse.Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.sparse.addmm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sparse.addmm(mat, mat1, mat2, *, beta=1., alpha=1.) -> Tensor This function does exact same thing as :func:`torch.addmm` in the forward, except that it supports backward for sparse COO matrix :attr... |
| `torch.sparse.as_sparse_gradcheck` | ❓ | ❓ | ❓ | ❓ | 🔴 | `gradcheck` | `Any` | Decorate function, to extend gradcheck for sparse tensors. Decorator for torch.autograd.gradcheck or its functools.partial variants that extends the gradcheck function with support to input functio... |
| `torch.sparse.check_sparse_tensor_invariants` | ❓ | ❓ | ❓ | ❓ | 🔴 | `enable` | `Any` | A tool to control checking sparse tensor invariants. The following options exists to manage sparsr tensor invariants checking in sparse tensor construction: 1. Using a context manager: .. code:: py... |
| `torch.sparse.log_softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sparse.log_softmax(input, dim, *, dtype=None) -> Tensor Applies a softmax function followed by logarithm. See :class:`~torch.sparse.softmax` for more details. Args: input (Tensor): input dim (int):... |
| `torch.sparse.mm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Performs a matrix multiplication of the sparse matrix :attr:`mat1` and the (sparse or strided) matrix :attr:`mat2`. Similar to :func:`torch.mm`, if :attr:`mat1` is a :math:`(n \times m)` tensor, :a... |
| `torch.sparse.sampled_addmm` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sparse.sampled_addmm(input, mat1, mat2, *, beta=1., alpha=1., out=None) -> Tensor Performs a matrix multiplication of the dense matrices :attr:`mat1` and :attr:`mat2` at the locations specified by ... |
| `torch.sparse.softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sparse.softmax(input, dim, *, dtype=None) -> Tensor Applies a softmax function. Softmax is defined as: :math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}` where :math:`i, j` run over s... |
| `torch.sparse.spdiags` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sparse.spdiags(diagonals, offsets, shape, layout=None) -> Tensor Creates a sparse 2D tensor by placing the values from rows of :attr:`diagonals` along specified diagonals of the output The :attr:`o... |
| `torch.sparse.spsolve` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sparse.spsolve(input, other, *, left=True) -> Tensor Computes the solution of a square system of linear equations with a unique solution. Its purpose is similar to :func:`torch.linalg.solve`, excep... |
| `torch.sparse.sum` | ❓ | ❓ | ❓ | ❓ | 🔴 | `input, dim, dtype` | `<class 'torch.Tensor'>` | Return the sum of each row of the given sparse tensor. Returns the sum of each row of the sparse tensor :attr:`input` in the given dimensions :attr:`dim`. If :attr:`dim` is a list of dimensions, re... |
| `torch.sparse.to_sparse_semi_structured` | ❓ | ❓ | ❓ | ❓ | 🔴 | `original_tensor, transposed` | `<class 'torch.sparse.semi_structured.SparseSemiStructuredTensor'>` | This function converts a dense tensor into a sparse semi-structured tensor. It will return a SparseSemiStructuredTensor, a subclass of torch.Tensor. This function will check to ensure the dense ten... |
| | | | | | | | | |
| 🟦 SPECIAL_FUNCTIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.special.Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.special.airy_ai` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | airy_ai(input, *, out=None) -> Tensor Airy function :math:`\text{Ai}\left(\text{input}\right)`. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output tensor. |
| `torch.special.bessel_j0` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bessel_j0(input, *, out=None) -> Tensor Bessel function of the first kind of order :math:`0`. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output tensor. |
| `torch.special.bessel_j1` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bessel_j1(input, *, out=None) -> Tensor Bessel function of the first kind of order :math:`1`. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output tensor. |
| `torch.special.bessel_y0` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bessel_y0(input, *, out=None) -> Tensor Bessel function of the second kind of order :math:`0`. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output tensor. |
| `torch.special.bessel_y1` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | bessel_y1(input, *, out=None) -> Tensor Bessel function of the second kind of order :math:`1`. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the output tensor. |
| `torch.special.chebyshev_polynomial_t` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | chebyshev_polynomial_t(input, n, *, out=None) -> Tensor Chebyshev polynomial of the first kind :math:`T_{n}(\text{input})`. If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{i... |
| `torch.special.chebyshev_polynomial_u` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | chebyshev_polynomial_t(input, n, *, out=None) -> Tensor Chebyshev polynomial of the second kind :math:`U_{n}(\text{input})`. If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`2 \tim... |
| `torch.special.chebyshev_polynomial_v` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | chebyshev_polynomial_v(input, n, *, out=None) -> Tensor Chebyshev polynomial of the third kind :math:`V_{n}^{\ast}(\text{input})`. Args: input (Tensor): the input tensor. n (Tensor): Degree of the ... |
| `torch.special.chebyshev_polynomial_w` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | chebyshev_polynomial_w(input, n, *, out=None) -> Tensor Chebyshev polynomial of the fourth kind :math:`W_{n}^{\ast}(\text{input})`. Args: input (Tensor): the input tensor. n (Tensor): Degree of the... |
| `torch.special.digamma` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | digamma(input, *, out=None) -> Tensor Computes the logarithmic derivative of the gamma function on `input`. .. math:: \digamma(x) = \frac{d}{dx} \ln\left(\Gamma\left(x\right)\right) = \frac{\Gamma'... |
| `torch.special.entr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | entr(input, *, out=None) -> Tensor Computes the entropy on :attr:`input` (as defined below), elementwise. .. math:: \begin{align} \text{entr(x)} = \begin{cases} -x * \ln(x) & x > 0 \\ 0 & x = 0.0 \... |
| `torch.special.erf` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | erf(input, *, out=None) -> Tensor Computes the error function of :attr:`input`. The error function is defined as follows: .. math:: \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt A... |
| `torch.special.erfc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | erfc(input, *, out=None) -> Tensor Computes the complementary error function of :attr:`input`. The complementary error function is defined as follows: .. math:: \mathrm{erfc}(x) = 1 - \frac{2}{\sqr... |
| `torch.special.erfcx` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | erfcx(input, *, out=None) -> Tensor Computes the scaled complementary error function for each element of :attr:`input`. The scaled complementary error function is defined as follows: .. math:: \mat... |
| `torch.special.erfinv` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | erfinv(input, *, out=None) -> Tensor Computes the inverse error function of :attr:`input`. The inverse error function is defined in the range :math:`(-1, 1)` as: .. math:: \mathrm{erfinv}(\mathrm{e... |
| `torch.special.exp2` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | exp2(input, *, out=None) -> Tensor Computes the base two exponential function of :attr:`input`. .. math:: y_{i} = 2^{x_{i}} Args: input (Tensor): the input tensor. Keyword args: out (Tensor, option... |
| `torch.special.expit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | expit(input, *, out=None) -> Tensor Computes the expit (also known as the logistic sigmoid function) of the elements of :attr:`input`. .. math:: \text{out}_{i} = \frac{1}{1 + e^{-\text{input}_{i}}}... |
| `torch.special.expm1` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | expm1(input, *, out=None) -> Tensor Computes the exponential of the elements minus 1 of :attr:`input`. .. math:: y_{i} = e^{x_{i}} - 1 .. note:: This function provides greater precision than exp(x)... |
| `torch.special.gammainc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | gammainc(input, other, *, out=None) -> Tensor Computes the regularized lower incomplete gamma function: .. math:: \text{out}_{i} = \frac{1}{\Gamma(\text{input}_i)} \int_0^{\text{other}_i} t^{\text{... |
| `torch.special.gammaincc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | gammaincc(input, other, *, out=None) -> Tensor Computes the regularized upper incomplete gamma function: .. math:: \text{out}_{i} = \frac{1}{\Gamma(\text{input}_i)} \int_{\text{other}_i}^{\infty} t... |
| `torch.special.gammaln` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | gammaln(input, *, out=None) -> Tensor Computes the natural logarithm of the absolute value of the gamma function on :attr:`input`. .. math:: \text{out}_{i} = \ln \Gamma(|\text{input}_{i}|) Args: in... |
| `torch.special.hermite_polynomial_h` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hermite_polynomial_h(input, n, *, out=None) -> Tensor Physicist's Hermite polynomial :math:`H_{n}(\text{input})`. If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{input}` is ... |
| `torch.special.hermite_polynomial_he` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | hermite_polynomial_he(input, n, *, out=None) -> Tensor Probabilist's Hermite polynomial :math:`He_{n}(\text{input})`. If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{input}`... |
| `torch.special.i0` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | i0(input, *, out=None) -> Tensor Computes the zeroth order modified Bessel function of the first kind for each element of :attr:`input`. .. math:: \text{out}_{i} = I_0(\text{input}_{i}) = \sum_{k=0... |
| `torch.special.i0e` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | i0e(input, *, out=None) -> Tensor Computes the exponentially scaled zeroth order modified Bessel function of the first kind (as defined below) for each element of :attr:`input`. .. math:: \text{out... |
| `torch.special.i1` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | i1(input, *, out=None) -> Tensor Computes the first order modified Bessel function of the first kind (as defined below) for each element of :attr:`input`. .. math:: \text{out}_{i} = \frac{(\text{in... |
| `torch.special.i1e` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | i1e(input, *, out=None) -> Tensor Computes the exponentially scaled first order modified Bessel function of the first kind (as defined below) for each element of :attr:`input`. .. math:: \text{out}... |
| `torch.special.laguerre_polynomial_l` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | laguerre_polynomial_l(input, n, *, out=None) -> Tensor Laguerre polynomial :math:`L_{n}(\text{input})`. If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{input}` is returned. ... |
| `torch.special.legendre_polynomial_p` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | legendre_polynomial_p(input, n, *, out=None) -> Tensor Legendre polynomial :math:`P_{n}(\text{input})`. If :math:`n = 0`, :math:`1` is returned. If :math:`n = 1`, :math:`\text{input}` is returned. ... |
| `torch.special.log1p` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | log1p(input, *, out=None) -> Tensor Alias for :func:`torch.log1p`. |
| `torch.special.log_ndtr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | log_ndtr(input, *, out=None) -> Tensor Computes the log of the area under the standard Gaussian probability density function, integrated from minus infinity to :attr:`input`, elementwise. .. math::... |
| `torch.special.log_softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | log_softmax(input, dim, *, dtype=None) -> Tensor Computes softmax followed by a logarithm. While mathematically equivalent to log(softmax(x)), doing these two operations separately is slower and nu... |
| `torch.special.logit` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logit(input, eps=None, *, out=None) -> Tensor Returns a new tensor with the logit of the elements of :attr:`input`. :attr:`input` is clamped to [eps, 1 - eps] when eps is not None. When eps is None... |
| `torch.special.logsumexp` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | logsumexp(input, dim, keepdim=False, *, out=None) Alias for :func:`torch.logsumexp`. |
| `torch.special.modified_bessel_i0` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | modified_bessel_i0(input, *, out=None) -> Tensor Modified Bessel function of the first kind of order :math:`0`. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the out... |
| `torch.special.modified_bessel_i1` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | modified_bessel_i1(input, *, out=None) -> Tensor Modified Bessel function of the first kind of order :math:`1`. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the out... |
| `torch.special.modified_bessel_k0` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | modified_bessel_k0(input, *, out=None) -> Tensor Modified Bessel function of the second kind of order :math:`0`. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the ou... |
| `torch.special.modified_bessel_k1` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | modified_bessel_k1(input, *, out=None) -> Tensor Modified Bessel function of the second kind of order :math:`1`. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the ou... |
| `torch.special.multigammaln` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | multigammaln(input, p, *, out=None) -> Tensor Computes the `multivariate log-gamma function <https://en.wikipedia.org/wiki/Multivariate_gamma_function>`_ with dimension :math:`p` element-wise, give... |
| `torch.special.ndtr` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ndtr(input, *, out=None) -> Tensor Computes the area under the standard Gaussian probability density function, integrated from minus infinity to :attr:`input`, elementwise. .. math:: \text{ndtr}(x)... |
| `torch.special.ndtri` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | ndtri(input, *, out=None) -> Tensor Computes the argument, x, for which the area under the Gaussian probability density function (integrated from minus infinity to x) is equal to :attr:`input`, ele... |
| `torch.special.polygamma` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | polygamma(n, input, *, out=None) -> Tensor Computes the :math:`n^{th}` derivative of the digamma function on :attr:`input`. :math:`n \geq 0` is called the order of the polygamma function. .. math::... |
| `torch.special.psi` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | psi(input, *, out=None) -> Tensor Alias for :func:`torch.special.digamma`. |
| `torch.special.round` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | round(input, *, out=None) -> Tensor Alias for :func:`torch.round`. |
| `torch.special.scaled_modified_bessel_k0` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | scaled_modified_bessel_k0(input, *, out=None) -> Tensor Scaled modified Bessel function of the second kind of order :math:`0`. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, opt... |
| `torch.special.scaled_modified_bessel_k1` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | scaled_modified_bessel_k1(input, *, out=None) -> Tensor Scaled modified Bessel function of the second kind of order :math:`1`. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, opt... |
| `torch.special.shifted_chebyshev_polynomial_t` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | shifted_chebyshev_polynomial_t(input, n, *, out=None) -> Tensor Chebyshev polynomial of the first kind :math:`T_{n}^{\ast}(\text{input})`. Args: input (Tensor): the input tensor. n (Tensor): Degree... |
| `torch.special.shifted_chebyshev_polynomial_u` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | shifted_chebyshev_polynomial_u(input, n, *, out=None) -> Tensor Chebyshev polynomial of the second kind :math:`U_{n}^{\ast}(\text{input})`. Args: input (Tensor): the input tensor. n (Tensor): Degre... |
| `torch.special.shifted_chebyshev_polynomial_v` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | shifted_chebyshev_polynomial_v(input, n, *, out=None) -> Tensor Chebyshev polynomial of the third kind :math:`V_{n}^{\ast}(\text{input})`. Args: input (Tensor): the input tensor. n (Tensor): Degree... |
| `torch.special.shifted_chebyshev_polynomial_w` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | shifted_chebyshev_polynomial_w(input, n, *, out=None) -> Tensor Chebyshev polynomial of the fourth kind :math:`W_{n}^{\ast}(\text{input})`. Args: input (Tensor): the input tensor. n (Tensor): Degre... |
| `torch.special.sinc` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | sinc(input, *, out=None) -> Tensor Computes the normalized sinc of :attr:`input.` .. math:: \text{out}_{i} = \begin{cases} 1, & \text{if}\ \text{input}_{i}=0 \\ \sin(\pi \text{input}_{i}) / (\pi \t... |
| `torch.special.softmax` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | softmax(input, dim, *, dtype=None) -> Tensor Computes the softmax function. Softmax is defined as: :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}` It is applied to all slices alo... |
| `torch.special.spherical_bessel_j0` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | spherical_bessel_j0(input, *, out=None) -> Tensor Spherical Bessel function of the first kind of order :math:`0`. Args: input (Tensor): the input tensor. Keyword args: out (Tensor, optional): the o... |
| `torch.special.xlog1py` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | xlog1py(input, other, *, out=None) -> Tensor Computes ``input * log1p(other)`` with the following cases. .. math:: \text{out}_{i} = \begin{cases} \text{NaN} & \text{if } \text{other}_{i} = \text{Na... |
| `torch.special.xlogy` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | xlogy(input, other, *, out=None) -> Tensor Computes ``input * log(other)`` with the following cases. .. math:: \text{out}_{i} = \begin{cases} \text{NaN} & \text{if } \text{other}_{i} = \text{NaN} \... |
| `torch.special.zeta` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | zeta(input, other, *, out=None) -> Tensor Computes the Hurwitz zeta function, elementwise. .. math:: \zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x} Args: input (Tensor): the input tensor cor... |
| | | | | | | | | |
| 🟦 STORAGE_MANAGEMENT | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.storage.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch.storage.Storage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.storage.TypeVar` | ❓ | ❓ | ❓ | ❓ | 🔴 | `name, constraints, bound, ...` | `Any` | Type variable. Usage:: T = TypeVar('T') # Can be anything A = TypeVar('A', str, bytes) # Must be str or bytes Type variables exist primarily for the benefit of static type checkers. They serve as t... |
| `torch.storage.TypedStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, wrap_storage, dtype, ...` | `Any` |  |
| `torch.storage.UntypedStorage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` |  |
| `torch.storage.cast` | ❓ | ❓ | ❓ | ❓ | 🔴 | `typ, val` | `Any` | Cast a value to a type. This returns the value unchanged. To the type checker this signals that the return value has the designated type, but at runtime we intentionally don't check anything (we wa... |
| | | | | | | | | |
| 🟦 TESTING_UTILITIES | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.testing.FileCheck` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.testing.assert_allclose` | ❓ | ❓ | ❓ | ❓ | 🔴 | `actual, expected, rtol, ...` | `None` | .. warning:: :func:`torch.testing.assert_allclose` is deprecated since ``1.12`` and will be removed in a future release. Please use :func:`torch.testing.assert_close` instead. You can find detailed... |
| `torch.testing.assert_close` | ❓ | ❓ | ❓ | ❓ | 🔴 | `actual, expected, allow_subclasses, ...` | `Any` | Asserts that ``actual`` and ``expected`` are close. If ``actual`` and ``expected`` are strided, non-quantized, real-valued, and finite, they are considered close if .. math:: \lvert \text{actual} -... |
| `torch.testing.make_tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `shape, dtype, device, ...` | `<class 'torch.Tensor'>` | Creates a tensor with the given :attr:`shape`, :attr:`device`, and :attr:`dtype`, and filled with values uniformly drawn from ``[low, high)``. If :attr:`low` or :attr:`high` are specified and are o... |
| | | | | | | | | |
| 🟦 TYPE_SYSTEM | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.types.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch.types.DispatchKey` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Members: Undefined CompositeExplicitAutogradNonFunctional CompositeExplicitAutograd CompositeImplicitAutogradNestedTensor CompositeImplicitAutograd AutogradNestedTensor AutogradOther Autograd Conju... |
| `torch.types.IO` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | Generic base class for TextIO and BinaryIO. This is an abstract, generic version of the return of open(). NOTE: This does not distinguish between the different possible classes (text vs. binary, re... |
| `torch.types.Sequence` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` | All the operations on a read-only sequence. Concrete subclasses must override __new__ or __init__, __getitem__, and __len__. |
| `torch.types.Size` | ❓ | ❓ | ❓ | ❓ | 🔴 | `iterable` | `Any` | Built-in immutable sequence. If no argument is given, the constructor returns an empty tuple. If iterable is specified the tuple is initialized from iterable's items. If the argument is a tuple, th... |
| `torch.types.Storage` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| `torch.types.SymBool` | ❓ | ❓ | ❓ | ❓ | 🔴 | `node` | `Any` | Like a bool (including magic methods), but redirects all operations on the wrapped node. This is used in particular to symbolically record operations in the symbolic shape workflow. Unlike regular ... |
| `torch.types.SymFloat` | ❓ | ❓ | ❓ | ❓ | 🔴 | `node` | `Any` | Like a float (including magic methods), but redirects all operations on the wrapped node. This is used in particular to symbolically record operations in the symbolic shape workflow. |
| `torch.types.SymInt` | ❓ | ❓ | ❓ | ❓ | 🔴 | `node` | `Any` | Like an int (including magic methods), but redirects all operations on the wrapped node. This is used in particular to symbolically record operations in the symbolic shape workflow. |
| `torch.types.Tensor` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `Any` |  |
| | | | | | | | | |
| 🟦 UTILITIES | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.utils.ThroughputBenchmark` | ❓ | ❓ | ❓ | ❓ | 🔴 | `module` | `Any` | This class is a wrapper around a c++ component throughput_benchmark::ThroughputBenchmark. This wrapper on the throughput_benchmark::ThroughputBenchmark component is responsible for executing a PyTo... |
| `torch.utils.generate_methods_for_privateuse1_backend` | ❓ | ❓ | ❓ | ❓ | 🔴 | `for_tensor, for_module, for_packed_sequence, ...` | `None` | Automatically generate attributes and methods for the custom backend after rename privateuse1 backend. In the default scenario, storage-related methods will not be generated automatically. When you... |
| `torch.utils.get_cpp_backtrace` | ❓ | ❓ | ❓ | ❓ | 🔴 | `frames_to_skip, maximum_number_of_frames` | `<class 'str'>` | Return a string containing the C++ stack trace of the current thread. Args: frames_to_skip (int): the number of frames to skip from the top of the stack maximum_number_of_frames (int): the maximum ... |
| `torch.utils.rename_privateuse1_backend` | ❓ | ❓ | ❓ | ❓ | 🔴 | `backend_name` | `None` | Rename the privateuse1 backend device to make it more convenient to use as a device name within PyTorch APIs. The steps are: (1) (In C++) implement kernels for various torch operations, and registe... |
| `torch.utils.set_module` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj, mod` | `Any` | Set the module attribute on a python object for a given object for nicer printing |
| `torch.utils.swap_tensors` | ❓ | ❓ | ❓ | ❓ | 🔴 | `t1, t2` | `Any` | This function swaps the content of the two Tensor objects. At a high level, this will make t1 have the content of t2 while preserving its identity. This will not work if t1 and t2 have different sl... |
| | | | | | | | | |
| 🟦 XPU_OPERATIONS | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
| `torch.xpu.Any` | ❓ | ❓ | ❓ | ❓ | 🔴 | `args, kwargs` | `Any` | Special type indicating an unconstrained type. - Any is compatible with every type. - Any assumed to have all methods. - All values assumed to be instances of Any. Note that all the above statement... |
| `torch.xpu.Event` | ❓ | ❓ | ❓ | ❓ | 🔴 | `enable_timing` | `Any` | Wrapper around a XPU event. XPU events are synchronization markers that can be used to monitor the device's progress, and to synchronize XPU streams. The underlying XPU events are lazily initialize... |
| `torch.xpu.Stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device, priority, kwargs` | `Any` | Wrapper around a XPU stream. A XPU stream is a linear sequence of execution that belongs to a specific device, independent from other streams. It supports with statement as a context manager to ens... |
| `torch.xpu.StreamContext` | ❓ | ❓ | ❓ | ❓ | 🔴 | `stream` | `Any` | Context-manager that selects a given stream. All XPU kernels queued within its context will be enqueued on a selected stream. Args: Stream (Stream): selected stream. This manager is a no-op if it's... |
| `torch.xpu.current_device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `` | `<class 'int'>` | Return the index of a currently selected device. |
| `torch.xpu.current_stream` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `<class 'torch.xpu.streams.Stream'>` | Return the currently selected :class:`Stream` for a given device. Args: device (torch.device or int, optional): selected device. Returns the currently selected :class:`Stream` for the current devic... |
| `torch.xpu.device` | ❓ | ❓ | ❓ | ❓ | 🔴 | `device` | `Any` | Context-manager that changes the selected device. Args: device (torch.device or int or str): device index to select. It's a no-op if this argument is a negative integer or ``None``. |
| `torch.xpu.device_of` | ❓ | ❓ | ❓ | ❓ | 🔴 | `obj` | `Any` | Context-manager that changes the current device to that of given object. You can use both tensors and storages as arguments. If a given object is not allocated on a XPU, this is a no-op. Args: obj ... |
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

---

## Implementation Status Tracking

### Overall Progress
- **Total Functions**: [To be calculated]
- **Completed**: [To be calculated]
- **In Progress**: [To be calculated]
- **Not Started**: [To be calculated]

### Priority Implementation Order
1. **Core Device Operations** (CUDA, MPS, CPU)
2. **Tensor Operations** (creation, manipulation, math)
3. **Neural Network Modules** (nn, functional)
4. **Optimization** (optim, autograd)
5. **Utilities** (utils, types, storage)

---

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

---

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

---

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

---

*Last Updated: 2024-01-XX*
*Total Functions: 3124*
*Implementation Status: 0/3124 Complete*
