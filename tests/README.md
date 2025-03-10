# TorchDevice Test Suite

This directory contains the test suite for the TorchDevice module. The tests ensure that TorchDevice correctly redirects CUDA calls to MPS (Metal Performance Shaders) and vice versa, with a fallback to CPU if neither is available.

## Test Structure

### common/ (Directory)

This directory contains shared test utilities used across all test modules:

- **test_utils.py**: Provides the `PrefixedTestCase` base class extending `unittest.TestCase` with:
  - Standard test setup and teardown procedures
  - Consistent test output formatting
  - Logging methods (`info`, `print_debug`, `warning`, `error`) for test diagnostics
  - Clear separation between test messages and TDLogger output

- **log_diff.py**: Contains utilities for test log capture and verification:
  - `setup_log_capture()`: Sets up logging capture for tests
  - `teardown_log_capture()`: Cleans up after tests
  - `diff_check()`: Compares captured logs with expected output files

## Test Files

### test_TorchDevice.py

This is the main test file for the TorchDevice module. It includes tests for:

- Device instantiation
- Tensor operations
- CUDA function calls
- Memory management
- CUDA events
- CUDA streams

### test_cuda_operations.py

This file contains comprehensive tests for CUDA operations, focusing on:

- Basic tensor operations
- Matrix multiplication
- CUDA stream operations
- CUDA event operations
- Stream and event interactions
- Multiple streams
- Stream synchronization and waiting mechanisms

These tests are derived from real-world usage patterns found in the `test_projects` directory to ensure that TorchDevice works correctly with typical PyTorch CUDA code.

### test_cpu_mps_operations.py

This file tests CPU and MPS operations specifically, ensuring:

- Proper tensor creation on both CPU and MPS devices
- Correct tensor operations on device-specific tensors
- Neural network operations on both devices
- Device conversion between CPU and MPS
- MPS device properties and behavior

### test_submodule.py

This file defines the `ModelTrainer` class, which is used in tests to simulate training operations while ensuring device compatibility.

### test_tdlogger/ (Directory)

This directory contains tests for the TDLogger module, which is responsible for logging device operations and redirections. See the [TDLogger Test Suite README](test_tdlogger/README.md) for more details.

The TDLogger tests include:
- Basic logging functionality
- Nested function calls
- Device operations
- Utility functions for log capture and comparison

## Running Tests

To run all tests and install the TorchDevice package:

```bash
python tests/run_tests_and_install.py
```

This script will:
1. Run all tests in the tests directory
2. If tests pass, build the package
3. Install the package in development mode

To run only tests without building and installing:

```bash
python tests/run_tests_and_install.py --test-only
```

To update expected output files for tests that use diff checking:

```bash
python tests/run_tests_and_install.py --update-expected
```

To run individual test files:

```bash
python tests/run_tests_and_install.py --test-only tests/test_TorchDevice.py
python tests/run_tests_and_install.py --test-only tests/test_cuda_operations.py
python tests/run_tests_and_install.py --test-only tests/test_tdlogger/test_basic.py
```

## Test Coverage

The test suite covers:

- Basic functionality: Device detection, tensor creation, and operations
- CUDA events: Creation, recording, synchronization, and timing
- CUDA streams: Creation, operations, synchronization, and context management
- Error handling: Proper fallback to available devices
- Real-world scenarios: Tests derived from typical PyTorch CUDA usage patterns
- Logging: Capturing and verifying log messages for device operations and redirections

## Adding New Tests

When adding new tests:

1. Extend the `PrefixedTestCase` class from `common/test_utils.py` for consistent test behavior
2. Use the logging methods provided by `PrefixedTestCase` for test diagnostics
3. Utilize `setup_log_capture()` and `diff_check()` from `common/log_diff.py` for log verification
4. Ensure tests are independent and can run in any order
5. Add appropriate assertions to verify functionality
6. Document the purpose of the test in docstrings
7. If adding a new test file, follow the naming convention `test_*.py`
