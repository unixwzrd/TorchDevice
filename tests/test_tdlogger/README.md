# TDLogger Test Suite

This directory contains tests for the TDLogger module, which is responsible for logging device operations and redirections in TorchDevice.

## Test Utilities

These tests use common utilities from the `tests/common/` directory:

### common/testing_utils.py

Provides the `PrefixedTestCase` base class with:

- Standard test setup and teardown procedures
- Consistent test output formatting
- Logging methods for test diagnostics
- Clear separation between test messages and TDLogger output

### common/log_diff.py

Provides log capture and verification utilities:

- `diff_check()`: Compares captured logs with expected output files
- `setup_log_capture()`: Sets up logging for tests
- `teardown_log_capture()`: Cleans up after tests

These utilities make it easy to write consistent tests that verify log output against expected results.

## Test Files

### test_basic.py

Tests basic logging functionality with TorchDevice, including:

- Capturing log output during tensor operations
- Verifying log messages against expected output
- Testing vector operations

### test_nested_calls.py

Tests logging of nested function calls with TorchDevice, ensuring:

- Correct caller information is captured
- Proper logging of redirections across multiple function levels
- Accurate tracking of the call stack

### test_device_operations.py

Tests various device operations with TorchDevice, including:

- Tensor creation on different devices
- Tensor movement between devices
- CUDA function calls and their logging

### test_device_utils.py

Tests device utility functions with TorchDevice, including:

- Device properties retrieval
- Memory management functions
- Stream and event interactions

### test_tensor_operations.py

Tests tensor operations with TorchDevice, including:

- Arithmetic operations on tensors
- Tensor reshaping operations
- Tensor indexing and slicing

## Running Tests

To run all TDLogger tests and update the expected output files:

```bash
python run_tests_and_install.py --update-expected tests/test_tdlogger/
```

To run the tests without updating the expected output files:

```bash
python run_tests_and_install.py --test-only tests/test_tdlogger/
```

To run a specific test file:

```bash
python run_tests_and_install.py --test-only tests/test_tdlogger/test_basic.py
```

## Adding New Tests

When adding new tests for TDLogger:

1. Extend the `PrefixedTestCase` class from `common/testing_utils.py`
2. Use `setup_log_capture()` and `teardown_log_capture()` from `common/log_diff.py`
3. Capture log output during test execution
4. Use `diff_check()` to compare the captured output with expected results
5. Create expected output files by running tests with the `--update-expected` flag

Example:

```python
from common.testing_utils import PrefixedTestCase
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture

class TestMyFeature(PrefixedTestCase):
    def setUp(self):
        # Call parent setUp which sets up logging
        super().setUp()
        
        # Set up log capture with a unique test name
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)
        
    def tearDown(self):
        # Clean up logging
        teardown_log_capture(self.log_capture)
        
        # Call parent tearDown which cleans up the test
        super().tearDown()
        
    def test_my_feature(self):
        # Print test info
        self.info("Testing my feature")
        
        # Run operations that generate logs
        # ...
        
        # Verify results
        self.info("Tests completed successfully")
        
        # Check the log output against expected
        diff_check(self.log_capture)
