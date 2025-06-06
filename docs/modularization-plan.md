# TorchDevice Modularization Plan

**Conventions:**

- Internal helper functions in modules use a leading underscore (e.g., _max_memory_reserved) to distinguish them from public API functions.
- Public API functions (those patched onto torch or torch.cuda) do not use the underscore and are thin wrappers around the internal helpers.
- All patches should be registered in core/patch.py for centralized management.
- No function names should be changed during migration to maintain compatibility.

This checklist outlines the steps for modularizing the TorchDevice codebase. Each phase should be completed and tested before moving to the next.

## Phase 1: Core Infrastructure ✓

- [x] Core Device Management (from TorchDevice.original/TorchDevice.py)
  - [x] Move device detection to core/device.py
  - [x] Move patch orchestration to core/patch.py
  - [x] Verify core/logger.py functionality

## Phase 2: Device Operations ✓

- [x] Device-Specific Operations (from TorchDevice.original/TorchDevice.py)
  - [x] Move CUDA operations to ops/device/cuda.py
  - [x] Move MPS operations to ops/device/mps.py
  - [x] Move CPU operations to ops/device/cpu.py

## Phase 3: Random Number Generation

- [x] Random Operations (from TorchDevice.original/device/random.py and ops/device/cuda.py)
  - [x] Centralize RNG state management and patching for `torch`, `torch.cuda`, and `torch.mps` into ops/random/generators.py
  - [x] Move distribution functions to ops/random/distributions.py
  - [x] Consolidate seed management functions within ops/random/generators.py
  - [x] Ensure all device-specific RNG operations (especially CUDA-related) are handled by ops/random/generators.py, removing stubs from ops/device/cuda.py

## Phase 4: Memory Management

- [x] Memory Operations (from TorchDevice.original/device/memory.py)
  - [x] Memory allocation in ops/memory/management.py
  - [x] Memory tracking in ops/memory/stats.py
  - [x] Cache management functions
  - [x] Device memory helpers

## Phase 5: Neural Network Operations

- [x] Neural Network Components (from TorchDevice.original/device/nn.py)
  - [x] Basic operations in ops/nn/layers.py
  - [x] Attention mechanisms in ops/nn/attention.py
  - [x] Normalization in ops/nn/normalization.py
  - [x] Activation functions in ops/nn/activation.py
  - [x] Weight initialization in ops/nn/init.py

## Phase 6: Stream and Event Handling

- [x] Stream Operations (from TorchDevice.original/device/streams.py)
  - [x] CUDA streams in ops/streams/cuda.py
  - [x] MPS streams in ops/streams/mps.py
  - [x] Synchronization in ops/streams/synchronize.py

## Phase 7: Utilities

- [x] Utility Functions (from TorchDevice.original/utils/)
  - [x] Compilation utilities in utils/compile.py
  - [x] Type handling in utils/type_utils.py
  - [x] Device utilities in utils/device_utils.py
  - [x] Error handling in utils/error_handling.py

## Additional Functions to Consider

- [x] Tensor Creation Functions
  - [x] Unified tensor_creation_wrapper relocated to core/patch.py for centralized use
  - [x] Device-specific tensor creation using unified wrapper
- [x] Autograd Functions
  - [x] Gradient computation (handled via AMP module)
  - [x] Backward hooks (no specific patches identified beyond device placement)
- [x] Optimizer Functions
  - [x] Device-aware parameter updates (handled by existing device management)
  - [x] State management (handled by existing device management)

## Testing Strategy

For each phase:

1. Create/update unit tests before migration
2. Move code without modification
3. Update imports in new location
4. Run tests to verify functionality
5. Only fix critical issues that prevent tests from passing

## Documentation Updates

- [ ] Update module docstrings
- [ ] Update README.md with new structure
- [ ] Update CHANGELOG.md with migration status
- [ ] Create/update API documentation

---

**Instructions:**

- Complete each phase in order
- Run all tests after each component migration
- Document any issues or needed fixes
- Do not modify function names or signatures
- Use TorchDevice.original as reference implementation

## Structure of Modules

- preserve all original Functions
- our new and override or redirect functions
- and apply_patches function at the end which is called during initialization of the Modules
  - and is also called when torch is loaded but has fencing in it to prevent double patching.

**Current Focus:**

- **Additional Functions / Testing**
