# TorchDevice Project TODO List

## TorchDevice.py Refactoring

- **Modularize TorchDevice Core Functions**
  - Split the large single-file implementation into logical modules:
    - Core device detection and management
    - CUDA function mocking
    - Stream and Event handling
    - Tensor operations interception
    - PyTorch integration utilities
  - Create proper module structure with appropriate imports
  - Ensure backward compatibility during transition

- **Fix Type Annotations and Linter Errors**
  - Address all the linter errors in TorchDevice.py
  - Add proper type annotations throughout the codebase
  - Fix compatibility with static type checkers
  - Maintain dynamic behavior while improving static analysis

- **CPU Override Beta Release Preparation**
  - ✅ Implement special 'cpu:-1' syntax for forcing CPU usage
  - ✅ Add global _cpu_override flag to track explicit CPU selection
  - ✅ Update device redirection logic to respect CPU override
  - ✅ Create dedicated tests for CPU override functionality
  - ✅ Update documentation and add examples
  - Test on different PyTorch versions to ensure compatibility
  - Test on different hardware configurations (CUDA, MPS, CPU-only)
  - Add more edge case tests for device indices and error conditions
  - Verify mixed precision operations work correctly with CPU override
  - Consider implementing runtime toggle for CPU override in future release
  - Investigate performance impact of CPU override vs. native CPU operations
  - Prepare beta release announcement and feedback mechanism

- **Improve Class Structure**
  - Review inheritance patterns for Stream and Event classes
  - Ensure proper inheritance from PyTorch base classes
  - Consolidate duplicate code in device handling
  - Improve encapsulation of internal state

- **Enhance Error Handling**
  - Add more robust error handling for device operations
  - Provide clearer error messages for device compatibility issues
  - Add fallback mechanisms for unsupported operations
  - Improve debugging information during redirection failures

## Logger Improvements

- **Simplify the Frame Inspection Logic**
  - Review the current implementation of `get_caller_info()`
  - Identify opportunities for simplification
  - Implement a more direct approach to find the caller frame

- **Implement Caching for Performance**
  - Identify functions that are called frequently with the same arguments
  - Add caching for these functions (e.g., using functools.lru_cache)
  - Measure the performance impact of these changes

- **Use Context Managers for Temporary State**
  - Identify operations that temporarily modify state
  - Implement context managers for these operations
  - Ensure proper cleanup even in case of exceptions

- **Implement Lazy Formatting for Log Messages**
  - Identify places where f-strings are used for messages that might be filtered out
  - Replace these with format strings that are only evaluated if the message will be logged
  - Verify that the changes don't affect the log output
  - Measure the performance impact of the change

- **Remove Unused Features and Code**
  - Identify unused code or overly complex sections
  - Simplify or remove these sections
  - Verify that the changes don't affect the functionality

## Device Handling Improvements

- ✅ **Implement CPU Override Feature**
  - ✅ Added special 'cpu:-1' device specification to force CPU usage
  - ✅ Implemented CPU override flag to ensure all subsequent operations respect explicit CPU selection
  - ✅ Enhanced device redirection logic to recognize and honor CPU override requests
  - ✅ Created dedicated test files for CPU, MPS, and override functionality
  - ✅ Simplified device handling logic for better maintainability
  - ✅ Added examples and updated documentation

- **Additional Device Handling Enhancements**
  - Allow toggling CPU override on/off during runtime
  - Provide finer-grained control over which operations respect the override
  - Optimize performance for CPU-specific operations
  - Add support for dynamic device allocation based on operation requirements

## Test Framework Improvements

- **Refine Test Utilities into Proper Modules**
  - Resolve import issues with common.log_diff and common.test_utils
  - Restructure as proper Python modules with __init__.py files
  - Update import paths in all test files
  - Add proper documentation for all test utility functions

- **Enhance Test Discoverability**
  - Improve test organization for better discoverability
  - Add test categorization (e.g., unit tests, integration tests)
  - Create a more flexible test runner with better filtering options

## Documentation

- **Create Developer Documentation**
  - Document the architecture and design decisions
  - Add detailed explanations of key components
  - Provide examples for common development tasks
  - Include troubleshooting guides for common issues

- **Improve API Documentation**
  - Document all public interfaces
  - Add examples for all major features
  - Include type annotations for better IDE support

## Known Issues

- TorchDevice.py has several linter errors that should be addressed in a future refactoring
- CPU override may not work with all third-party extensions to PyTorch
- The override remains active for the entire Python session; a future enhancement could allow toggling it on/off 