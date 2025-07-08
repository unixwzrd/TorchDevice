# TorchDevice Test Suite

## Overview

The test suite is organized to mirror the main project structure, with dedicated test modules for each component. The tests use Python's unittest framework with our custom `PrefixedTestCase` base class for consistent logging and output validation.

## Test Directory Structure

```
tests/
├── core/                   # Tests for core functionality
│   ├── test_device.py     # Device handling tests
│   ├── test_patch.py      # Patching mechanism tests
│   └── test_logger.py     # Logging system tests
│
├── ops/                   # Tests for operations
│   ├── memory/           # Memory management tests
│   ├── nn/               # Neural network tests
│   ├── random/           # Random number generation tests
│   ├── streams/          # Stream handling tests
│   ├── events/           # Event handling tests
│   ├── autograd/         # Automatic differentiation tests
│   └── optim/            # Optimization tests
│
├── utils/                # Tests for utilities
│   ├── test_compile.py  # Compilation utility tests
│   ├── test_profiling.py # Profiling tool tests
│   └── test_type_utils.py # Type utility tests
│
├── common/               # Shared test infrastructure
│   ├── testing_utils.py    # Common test utilities
│   └── log_diff.py      # Log comparison tools
│
└── integration/         # Integration tests
    ├── test_models.py  # Full model tests
    └── test_workflows.py # Common workflow tests
```

## Running Tests

### Basic Usage

```bash
# Run all tests
python run_tests_and_install.py

# Run specific test module
python run_tests_and_install.py --test-only tests/core/test_device.py

# Update expected outputs
python run_tests_and_install.py --test-only --update-expected tests/core/test_device.py
```

### Test Environment Variables

- `TORCHDEVICE_TEST_MODE`: Set to '1' to enable test mode
- `TORCHDEVICE_LOG_LEVEL`: Control log verbosity (default: 'INFO')
- `TORCHDEVICE_TEST_DEVICE`: Specify test device ('cuda', 'mps', or 'cpu')

## Writing Tests

### Base Test Class

Use `PrefixedTestCase` as your base class:

```python
from tests.common.testing_utils import PrefixedTestCase

class TestDeviceHandling(PrefixedTestCase):
    def setUp(self):
        super().setUp()
        # Your setup code

    def test_device_selection(self):
        # Your test code
```

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Regression Tests**: Prevent previously fixed bugs
4. **Performance Tests**: Verify performance characteristics

### Best Practices

1. **Test Organization**
   - One test class per module/feature
   - Clear test method names
   - Logical test grouping

2. **Test Coverage**
   - Test both success and failure cases
   - Include edge cases
   - Test type checking
   - Test error handling

3. **Test Independence**
   - Each test should be independent
   - Clean up resources in tearDown
   - Don't rely on test execution order

4. **Assertions**
   - Use specific assertions
   - Include meaningful error messages
   - Check both values and types

5. **Documentation**
   - Document test purpose
   - Document test requirements
   - Document expected behavior

### Example Test

```python
from tests.common.testing_utils import PrefixedTestCase
import torch
import TorchDevice

class TestDeviceSelection(PrefixedTestCase):
    """Tests for device selection and redirection."""

    def setUp(self):
        super().setUp()
        self.original_device = torch.device('cpu')

    def test_cuda_to_mps_redirection(self):
        """Test CUDA to MPS redirection when CUDA is unavailable."""
        # Arrange
        requested_device = torch.device('cuda')

        # Act
        actual_device = TorchDevice.get_device(requested_device)

        # Assert
        self.assertEqual(actual_device.type, 'mps',
                        "CUDA device should be redirected to MPS")
```

## Log Validation

The test suite includes automatic log validation:

1. Expected outputs are stored in `.expected` files
2. Test runs compare actual output with expected
3. Update expected outputs with `--update-expected`

## Adding New Tests

1. Create test file in appropriate directory
2. Inherit from `PrefixedTestCase`
3. Implement `setUp` and `tearDown` if needed
4. Add test methods
5. Generate expected outputs
6. Add to test discovery

## Continuous Integration

Tests are automatically run on:
- Pull requests
- Main branch commits
- Release tags

## Performance Testing

- Use `@performance_test` decorator
- Set baseline expectations
- Compare against previous results
- Account for hardware variations
