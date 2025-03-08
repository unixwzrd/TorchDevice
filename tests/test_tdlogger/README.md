# TDLogger Test Suite

This directory contains tests for the TDLogger module, which is responsible for logging device operations and redirections in TorchDevice.

## Test Utilities

### test_utils.py

This file provides common functionality for all TDLogger tests:

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

## Running Tests

To run all TDLogger tests and update the expected output files:

```bash
python tests/run_tests_and_install.py --update-expected tests/test_tdlogger/
```

To run the tests without updating the expected output files:

```bash
python tests/run_tests_and_install.py tests/test_tdlogger/
```

To run a specific test file:

```bash
python tests/run_tests_and_install.py tests/test_tdlogger/test_basic.py
```

## Adding New Tests

When adding new tests for TDLogger:

1. Import the utility functions from `test_utils.py`
2. Use `setup_log_capture()` and `teardown_log_capture()` in your test class
3. Capture log output during test execution
4. Use `diff_check()` to compare the captured output with expected results
5. Create expected output files by running tests with the `--update-expected` flag

Example:

```python
from test_utils import diff_check, setup_log_capture, teardown_log_capture

class TestMyFeature(unittest.TestCase):
    def setUp(self):
        # Set up logging
        result = setup_log_capture()
        self.logger = result[0]
        self.log_stream = result[1]
        self.log_handler = result[2]
        self.console_handler = result[3]
        self.original_handlers = result[4]
        self.original_level = result[5]
        
        # Define expected output file
        self.expected_output_file = Path(__file__).parent / f"{self._testMethodName}_expected.log"
        
    def tearDown(self):
        # Clean up logging
        teardown_log_capture(
            self.logger, 
            self.original_handlers, 
            self.original_level,
            [self.log_handler, self.console_handler]
        )
        
    def test_my_feature(self):
        # Run operations that generate logs
        # ...
        
        # Get captured log output
        captured_log = self.log_stream.getvalue()
        
        # Compare with expected output
        diff_check(captured_log, self.expected_output_file) 