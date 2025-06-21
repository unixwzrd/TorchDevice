# TorchDevice Project TODO List

> **Note:** As of the latest checkpoint, the project is at a stable state: all core logic is modularized, the CPU override feature is complete, and all core tests pass. Remaining items are for future enhancements, broader testing, and documentation expansion.

## TorchDevice.py Refactoring

- [x] **Modularize TorchDevice Core Functions**
  - [x] Split the large single-file implementation into logical modules:
    - Core device detection and management
    - CUDA function mocking
    - Stream and Event handling
    - Tensor operations interception
    - PyTorch integration utilities
  - [x] Create proper module structure with appropriate imports
  - [x] Ensure backward compatibility during transition

- [x] **Fix Type Annotations and Linter Errors**
  - [x] Address all the linter errors in TorchDevice.py (resolved by modularization)
  - [x] Add proper type annotations throughout the codebase
  - [x] Fix compatibility with static type checkers
  - [x] Maintain dynamic behavior while improving static analysis

- [x] **CPU Override Beta Release Preparation**
  - [x] Implement special 'cpu:-1' syntax for forcing CPU usage
  - [x] Add global _cpu_override flag to track explicit CPU selection
  - [x] Update device redirection logic to respect CPU override
  - [x] Create dedicated tests for CPU override functionality
  - [x] Update documentation and add examples
  - [ ] Test on different PyTorch versions to ensure compatibility
  - [ ] Test on different hardware configurations (CUDA, MPS, CPU-only)
  - [ ] Add more edge case tests for device indices and error conditions
  - [ ] Verify mixed precision operations work correctly with CPU override
  - [ ] Consider implementing runtime toggle for CPU override in future release
  - [ ] Investigate performance impact of CPU override vs. native CPU operations
  - [ ] Prepare beta release announcement and feedback mechanism

- [ ] **Improve Class Structure**
  - [ ] Review inheritance patterns for Stream and Event classes
  - [ ] Ensure proper inheritance from PyTorch base classes
  - [ ] Consolidate duplicate code in device handling
  - [ ] Improve encapsulation of internal state

- [ ] **Enhance Error Handling**
  - [ ] Add more robust error handling for device operations
  - [ ] Provide clearer error messages for device compatibility issues
  - [ ] Add fallback mechanisms for unsupported operations
  - [ ] Improve debugging information during redirection failures

## Logger Improvements

- [ ] **Simplify the Frame Inspection Logic**
  - [ ] Review the current implementation of `get_caller_info()`
  - [ ] Identify opportunities for simplification
  - [ ] Implement a more direct approach to find the caller frame

- [ ] **Implement Caching for Performance**
  - [ ] Identify functions that are called frequently with the same arguments
  - [ ] Add caching for these functions (e.g., using functools.lru_cache)
  - [ ] Measure the performance impact of these changes

- [ ] **Use Context Managers for Temporary State**
  - [ ] Identify operations that temporarily modify state
  - [ ] Implement context managers for these operations
  - [ ] Ensure proper cleanup even in case of exceptions

- [x] **Implement Lazy Formatting for Log Messages**
  - [x] Identify places where f-strings are used for messages that might be filtered out
  - [x] Replace these with format strings that are only evaluated if the message will be logged
  - [x] Verify that the changes don't affect the log output
  - [x] Measure the performance impact of the change

- [ ] **Remove Unused Features and Code**
  - [ ] Identify unused code or overly complex sections
  - [ ] Simplify or remove these sections
  - [ ] Verify that the changes don't affect the functionality

- [ ] **Reduce Default Logger Verbosity**
  - [ ] Implement a log level configuration (e.g., via environment variable or function call).
  - [ ] Change the default log level to be less verbose (e.g., INFO or WARNING).
  - [ ] Ensure debug-level logging remains available for development.

## Device Handling Improvements

- [x] **Implement CPU Override Feature**
  - [x] Added special 'cpu:-1' device specification to force CPU usage
  - [x] Implemented CPU override flag to ensure all subsequent operations respect explicit CPU selection
  - [x] Enhanced device redirection logic to recognize and honor CPU override requests
  - [x] Created dedicated test files for CPU, MPS, and override functionality
  - [x] Simplified device handling logic for better maintainability
  - [x] Added examples and updated documentation

- [ ] **Additional Device Handling Enhancements**
  - [ ] Allow toggling CPU override on/off during runtime
  - [ ] Provide finer-grained control over which operations respect the override
  - [ ] Optimize performance for CPU-specific operations
  - [ ] Add support for dynamic device allocation based on operation requirements

## Test Automation & Documentation

- [x] **Create Test Automation Framework**
  - [x] Enhanced `generate_test_report.py` for better path resolution and clickable links.
  - [x] Updated `test_automation/README.md` with setup and run instructions.
- [x] **Update Project Documentation**
  - [x] Added link to advanced test automation guide in main `README.md`.
  - [x] Updated `CONTRIBUTING.md` with instructions for running external project tests.

## Test Framework Improvements

- [ ] **Refine Test Utilities into Proper Modules**
  - [ ] Resolve import issues with common.log_diff and common.test_utils
  - [ ] Restructure as proper Python modules with __init__.py files
  - [ ] Update import paths in all test files
  - [ ] Add proper documentation for all test utility functions

- [ ] **Enhance Test Discoverability**
  - [ ] Improve test organization for better discoverability
  - [ ] Add test categorization (e.g., unit tests, integration tests)
  - [ ] Create a more flexible test runner with better filtering options

## Documentation

- [x] **Create Developer Documentation**
  - [x] Document the architecture and design decisions
  - [x] Add detailed explanations of key components
  - [x] Provide examples for common development tasks
  - [x] Include troubleshooting guides for common issues

- [x] **Improve API Documentation**
  - [x] Document all public interfaces
  - [x] Add examples for all major features
  - [x] Include type annotations for better IDE support

## Known Issues

- TorchDevice.py has several linter errors that should be addressed in a future refactoring (mostly resolved by modularization)
- CPU override may not work with all third-party extensions to PyTorch
- The override remains active for the entire Python session; a future enhancement could allow toggling it on/off 