# CHANGELOG

## 0.5.2 - 2025-06-21

### Cleanup, Documentation, and Test Results

### Added

- **Test Automation Framework (`test_automation/`)**:
  - Enhanced `generate_test_report.py` script to improve test file path resolution and clickable links in the Markdown report.
  - Updated `test_automation/README.md` with instructions for setting up and running the Transformers test suite.
  - Updated [README](README.md) with a link to the advanced test automation guide.
  - Updated [CONTRIBUTING](CONTRIBUTING.md) with instructions for running external tests.

## 0.5.1 - 2025-06-20

### Improved

- **Test Reporting**:
  - Enhanced the `generate_test_report.py` script to make test file paths in the Markdown report clickable, improving navigation from the report directly to the source code.
  - Corrected relative link paths to ensure they resolve correctly from the report's location in `test_automation/reports/`.
- **Documentation**:
  - Updated `test_automation/README.md` with a new section detailing required system-level dependencies (Tesseract, image libraries) for running the full Transformers test suite.
  - Added a link in the main project `README.md` pointing to the advanced test automation guide for better discoverability.


## 0.4.2 - 2025-06-08

### Added

- **Test Automation Framework (`test_automation/`)**:
  - Introduced `run_transformer_tests.py`, a versatile script for automating test suite execution for Python projects with `TorchDevice` integration.
  - Key features:
    - Discovers `test_*.py` files within specified target directories (relative to project's `tests/` dir).
    - Executes tests using `python -m unittest -v` in isolated subprocesses.
    - Automatically sets `ACTIVATE_TORCH_DEVICE=1` environment variable to enable `TorchDevice`.
    - Dynamically configures `PYTHONPATH` to include `TorchDevice` and target project's `src/` directory.
    - Generates comprehensive logs:
      - Overall run log with a descriptive name (e.g., `<series>_<timestamp>.log`) in `test_automation/logs/`.
      - Individual test logs (STDOUT/STDERR) in `test_automation/logs/<project_name>/...`.
    - Configurable target project root (`--project_root`) and `TorchDevice` root (`--torchdevice_root`).
    - Designed for general use with Python projects adhering to common test structures.
  - Added `test_automation/README.md` detailing script usage, configuration, and logging.

## 2025-06-06 - 0.4.1 - Attention Test Refactoring

### Improved

- Refactored `tests/test_attention.py` to exclusively use standard PyTorch APIs (`torch.nn.functional.scaled_dot_product_attention`, `torch.nn.MultiheadAttention`, and `transformers.BertSelfAttention`) instead of internal TorchDevice replacement functions. This ensures tests accurately validate TorchDevice's transparent patching capabilities for attention mechanisms.
- Removed direct imports of TorchDevice internal functions from `tests/test_attention.py`.
- Ensured `import TorchDevice` is present in `tests/test_attention.py` to apply patches before test execution.

### Added

- Complete project restructuring with new modular architecture
  - New `core/` module for central device handling
  - New `ops/` module for operation-specific implementations
  - New `utils/` module for shared utilities
- Added comprehensive documentation in `docs/project_structure.md`
- Added new modules for enhanced functionality:
  - Events handling and synchronization
  - Autograd support
  - Optimization algorithms
  - Enhanced stream management

### Changed

- Reorganized existing code into logical modules
- Updated import paths to reflect new structure
- Consolidated patch functionality into core module
- Improved separation of concerns between modules
- Centralized `torch.cuda` RNG (Random Number Generation) patching logic into `TorchDevice/ops/random/generators.py`.
- Refactored `TorchDevice/ops/random/generators.py` to use `DeviceManager` and `hardware_info` for robust, device-aware RNG patching across `torch`, `torch.cuda`, and `torch.mps` namespaces.

  - **Core Patching Refinement**:
    - Further modularized core patching logic within `TorchDevice/core/`:
      - `device.py`: Confirmed `DeviceManager` handles `torch.device` and `torch.load` patching.
      - `tensors.py`: Streamlined for `torch.Tensor` method patching and tensor creation wrappers.
      - `modules.py`: Introduced for dedicated `torch.nn.Module` method patching.
      - `patch.py`: Updated to orchestrate the refined patching sequence from these modules.

### Removed

- Redundant patch implementations
- Deprecated compile-related files
- Removed redundant RNG stubs and patching logic from `TorchDevice/ops/device/cuda.py` as this is now fully handled by `ops/random/generators.py`.

## 2025-05-26 - 0.4.0 - Transformer Support and Attention Mechanisms

### Major Changes

- **Comprehensive Transformer Support**
  - Added full support for transformer model operations
  - Implemented scaled dot-product attention mechanism
  - Added multi-head attention support with BERT compatibility
  - Enhanced device handling for attention operations
  - Added proper type checking and device compatibility
  - Implemented automatic device redirection for attention ops

- **Neural Network Enhancements**
  - Fixed embedding operation issues and improved type safety
  - Enhanced tensor device and dtype management
  - Added proper error handling and fallbacks
  - Improved performance for common operations
  - Added comprehensive tensor type validation
  - Fixed embedding_renorm_ operation issues

- **Architecture Improvements**
  - Centralized neural network operations in `device/nn.py`
  - Added dedicated attention mechanism module in `device/attention.py`
  - Enhanced patch application system for better modularity
  - Improved device redirection for transformer operations
  - Added automatic patch registration for new modules

- **Testing and Validation**
  - Added comprehensive test suite for attention mechanisms
  - Enhanced BERT model compatibility tests
  - Added tests for embedding and linear operations
  - Improved test coverage for layer normalization
  - Added transformer model integration tests
  - Enhanced device compatibility test coverage

### Breaking Changes

- Attention mechanisms now enforce stricter type checking
- Some attention operations may require explicit device specifications
- Neural network operations require proper dtype specifications
- Device handling follows stricter validation rules

### Known Issues

- Some specialized CUDA attention operations may not have direct MPS equivalents
- Performance implications when using attention mechanisms on CPU fallback
- Certain CUDA-specific optimizations may not be available on MPS
- Some transformer operations may require additional memory on MPS devices

### Next Steps

- Optimization of attention operations for different hardware
- Support for more transformer architectures
- Enhanced memory management for large models
- Performance profiling and improvements
- Implementation of additional CUDA-specific features
- Enhanced error reporting and diagnostics

### Testing Notes

We need testers to validate the following scenarios:

1. Large transformer model inference
2. Multi-head attention performance
3. Mixed-precision training with attention
4. Cross-attention mechanisms
5. Causal attention masking
6. Device switching during model execution
7. Memory usage patterns on different devices
8. Performance comparison between CUDA and MPS

Please report any issues or unexpected behavior through the issue tracker.

### Package Updates

- Version updated to 0.4.0 in all relevant files
- Added transformers>=4.30.0 as a dependency
- Updated package classifiers to reflect beta status
- Enhanced package discovery with find_packages()
- Added new module exports in **init**.py

## 2025-05-24 - 0.2.0 - Neural Network Operations and Device Handling Overhaul

### Major Changes

- **Complete Neural Network Operations Refactoring**
  - Centralized all neural network operations in dedicated `device/nn.py` module
  - Added comprehensive type safety and device compatibility checks
  - Implemented proper tensor dtype handling across operations
  - Added support for embedding, linear, and layer normalization operations
  - Fixed critical issues with embedding operations on MPS devices

- **Enhanced Device Management**
  - Improved device redirection logic for better compatibility
  - Added robust type conversion handling for tensor operations
  - Fixed device-specific normalization issues
  - Enhanced memory management for tensor operations

- **Modular Architecture**
  - Reorganized codebase into logical modules for better maintainability
  - Separated device-specific operations into dedicated modules
  - Implemented helper utilities for common tensor operations
  - Improved code reusability and reduced duplication

- **Testing Infrastructure**
  - Added comprehensive tests for neural network operations
  - Enhanced test coverage for device handling
  - Improved test reliability and reproducibility
  - Added transformer model integration tests

### Breaking Changes

- Neural network operations now enforce stricter type checking
- Device handling may require explicit dtype specifications in some cases
- Embedding operations now handle normalization differently

### Known Issues

- Some CUDA-specific operations may not have full MPS equivalents
- Performance implications when falling back to CPU for unsupported operations

### Next Steps

- Implementation of attention mechanisms
- Support for more neural network operations
- Enhanced error handling and diagnostics
- Performance optimizations for device-specific operations

### Testing Notes

We need testers to validate the following scenarios:

1. Transformer model inference on MPS devices
2. Large-scale embedding operations
3. Mixed-precision training workflows
4. Multi-device tensor operations

Please report any issues or unexpected behavior through the issue tracker.

## 2025-03-16 - 0.1.1

### Interim Checkpoint Release

- **Modularization Complete:** All core logic is now modularized into dedicated modules under `TorchDevice/cuda/`.
- **CPU Override Feature Stable:** The `cpu:-1` override is fully implemented, documented, and tested.
- **All Core Tests Passing:** All unit and integration tests for the modularized codebase are passing as of this release.
- **Documentation Updated:** README and developer docs reflect the new structure and features.
- **Next Focus:** Running and validating all example/demo scripts in the `examples/` directory, and expanding user-facing features.

## 2025-03-15 - 0.0.5

### CPU Override Feature

- **Explicit CPU Device Selection**
  - Added new `cpu:-1` device specification to force CPU usage regardless of available accelerators
  - Implemented CPU override flag to ensure all subsequent operations respect explicit CPU selection
  - Enhanced device redirection logic to recognize and honor CPU override requests
  - Simplified device handling logic for better maintainability and reliability

- **Testing Infrastructure Improvements**
  - Separated CPU and MPS tests into dedicated modules to prevent test interference:
    - `test_cpu_operations.py` for testing CPU-specific functionality
    - `test_mps_operations.py` for testing MPS-specific functionality
    - `test_cpu_override.py` for testing the new CPU override feature
  - Fixed device handling in tests to properly isolate test environments
  - Updated expected output files to accommodate new logging patterns
  - Enhanced test robustness with better device state management

- **Device Handling Logic Improvements**
  - Simplified `torch_device_replacement` function with clearer, more declarative logic
  - Enhanced handling of device indices for both string and separated parameter formats
  - Improved input validation to prevent invalid device specifications
  - Streamlined error handling for more robust operation with edge cases

## 2025-03-10 - 0.0.4

### TDLogger Improvements

- **Consolidated Duplicate Code**
  - Added new helper functions to centralize common operations:
    - `contains_important_message()` - Checks if a message contains important patterns that should always be logged
    - `is_setup_or_init()` - Identifies setup and initialization functions for better filtering
    - `format_message()` - Centralizes message formatting for consistency
  - Reorganized code structure for better clarity and maintainability

- **Optimized Message Filtering Logic**
  - Improved the `should_skip_message()` function with clearer, more declarative logic
  - Created constant `IMPORTANT_MESSAGE_PATTERNS` to centralize patterns that should always be logged
  - Enhanced filtering to ensure critical messages are never skipped
  - Simplified complex conditional logic for better readability and maintainability

- **Improved Error Handling**
  - Added robust try/except blocks in `log_message()` to catch and handle exceptions
  - Added error reporting that directs error messages to stderr for easy debugging
  - Ensured exceptions in the logger won't propagate to the main application
  - Fixed edge cases that could cause unexpected behavior during logging

- **Test Framework Enhancements**
  - Improved `PrefixedTestCase` class with clear separation between test messages and TDLogger output
  - Fixed formatting inconsistencies in the test output for better readability
  - Updated expected output files to reflect new logger behavior
  - Enhanced error handling in test utilities

### Test Infrastructure Improvements

- **Created Common Test Utilities Directory**
  - Added `tests/common/` directory to house shared test infrastructure
  - Implemented `test_utils.py` with the `PrefixedTestCase` class providing standardized logging capabilities
  - Moved `log_diff.py` to common location for reuse across all test modules
  - Standardized test setup and teardown procedures across all tests
  - Improved test discoverability and organization

## 2025-03-08 - 0.0.3

### Logger Improvements

- Optimized the `_logged_messages` collection by replacing the set with a fixed-size deque
- This change improves memory management by automatically removing oldest entries when the collection is full
- Eliminates the need to clear the entire collection, providing more consistent duplicate prevention
- Consolidated duplicate code by extracting common logic into helper functions:
  - Added `is_test_environment()` to determine if running in a test environment
  - Added `is_internal_frame()` to identify frames that should be skipped
  - Added `should_skip_message()` to centralize message filtering logic
  - Added `is_setup_or_init()` to identify setup and initialization functions
- Improved caller identification to ensure log messages show the actual caller
- Enhanced message filtering to ensure important redirection messages are always logged

### Test Framework Improvements

- Created a robust test framework with a `PrefixedTestCase` class in `test_utils.py`
- Fixed test discovery and execution to ensure consistent environment variables
- Improved logging during tests with better context and error handling
- Enhanced expected output file management for more reliable test results
- Fixed issues with program name consistency in log messages during tests

## 2025-03-03 - 0.0.2

### Logging System Improvements

- Modularized logging: Moved logging functionality into its own module (`TDLogger.py`)
- Simplified logging interface:
  - Removed verbosity levels (LOG_VERBOSITY) in favor of a simpler on/off approach
  - Eliminated different message classes (warning, info, error) for a more streamlined logging experience
  - Consolidated all logging to use a single `log_message` function as the primary entry point
  - Removed redundant `log` function since the project hasn't been publicly released yet
- Improved caller tracking for more accurate log messages
- Optimized memory usage by limiting the size of the logged messages cache
- Enhanced test suite with expected output validation

## 2024-12-30 - 0.0.1

- Added note regarding building NumPy for Apple silicon.

## 2024-12-12 - Initial release

- Initial release
  - Need other to get involved and help test/make improvements.
