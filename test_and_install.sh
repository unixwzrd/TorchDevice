#!/bin/bash
# Simple wrapper script to run the tests, build, and install TorchDevice
#
# Usage:
#   ./test_and_install.sh           # Run tests, build, and install
#   ./test_and_install.sh --test-only  # Run only tests
#   ./test_and_install.sh --help    # Show help message

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Run the Python script with any arguments passed to this script
python "${SCRIPT_DIR}/tests/run_tests_and_install.py" "$@"

# Exit with the same status code as the Python script
exit $?
