# TorchDevice Modularization Plan

**Conventions:**
- Internal helper functions in modules (e.g., memory, random, unassigned) use a leading underscore (e.g., _max_memory_reserved) to distinguish them from public API functions.
- Public API functions (those patched onto torch or torch.cuda) do not use the underscore and are thin wrappers around the internal helpers.

This checklist outlines the steps for modularizing the TorchDevice codebase. Check off each item as it is completed.

- [x] Move device-aware tensor creation and RNG/seed patching logic to `TorchDevice/cuda/random.py`.

- [x] Move memory management and patching logic (e.g., memory allocation, caching, device memory helpers) to `TorchDevice/cuda/memory.py`.
    - Extract all memory-related helpers and patching from `TorchDevice.py`.
    - Ensure all memory patching is handled in the new module.
    - Add stubs/placeholders for future memory management features if needed.

- [x] Move CUDA stubs and mock logic to `TorchDevice/cuda/unassigned.py` (formerly planned as stubs.py).
    - Extract all CUDA stub/mock functions from `TorchDevice.py`.
    - Ensure compatibility with environments lacking CUDA support.

- [x] Create or update a central patch application module (`TorchDevice/cuda/patch.py`).
    - Ensure all patching (random, memory, unassigned) is orchestrated from this module.
    - Update imports and patch calls in `TorchDevice.py` and elsewhere as needed.
    - All patching is now centralized and dead code has been audited/removed.

- [x] Update all imports in `TorchDevice.py` and other modules to use the new modular structure.
    - Remove any redundant or duplicate code.

- [x] Update or create unit tests for each new module.
    - Ensure all moved logic is covered by tests.
    - Adjust or remove tests for deprecated/removed functionality.
    - All current tests pass after modularization (checkpoint commit recommended).

- [ ] Review and update documentation to reflect the new modular structure.
    - Update references in `docs/TorchDevice_Functions.md` and related docs.

---

**Instructions:**
- Work through each item in order, checking off as you complete each step.
- If you encounter additional modularization opportunities, add them to this checklist. 

Naming Convention:
- Internal helper functions use the same name as the public API but with a leading underscore
  (e.g., _memory_allocated for torch.cuda.memory_allocated).
- Public API functions (patched onto torch.cuda) do not use the underscore and are thin wrappers around the internal helpers.

---

**Next focus:**
- Run and fix all example/demo tests in the `examples/` directory.
- Add further functionality as needed for full coverage and user-facing features.