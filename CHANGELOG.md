# CHANGELOG

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

