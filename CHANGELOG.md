# CHANGELOG

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
