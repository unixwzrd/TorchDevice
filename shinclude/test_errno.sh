#!/usr/bin/env bash

# This script is used to test the "errno" function. It contains a placeholder function "run_tests" that can be used to add test cases.
# The "run_tests" function calls the "test_errno" function to test the "errno" function with different error names.
# The "assert_equals" function is used to assert equality between expected and actual values.
# The "test_errno" function takes an error name, expected error number, and expected error text as arguments.
# It calls the "errno" function with the error name and captures the actual error number and error text.
# The "assert_equals" function is then used to compare the expected and actual values.
# If the values match, a PASS message is printed. Otherwise, a FAIL message is printed along with the expected and actual values.

# Source in errno.sh from the proper directory
source "./errno.sh"

# Placeholder function for run_tests
run_tests() {
    # Add your test cases here
    echo "Running tests..."
    
    # Test the errno function
    test_errno "EACCES" "13" "Permission denied"
    test_errno "ENOENT" "2" "No such file or directory"
    test_errno "EINVAL" "22" "Invalid argument"
}

# Function to assert equality between two values
assert_equals() {
    local expected="$1"
    local actual="$2"
    local message="$3"

    if [[ "$expected" == "$actual" ]]; then
        echo "PASS: $message"
    else
        echo "FAIL: $message"
        echo "Expected: $expected"
        echo "Actual: $actual"
    fi
}

# Test the errno function
test_errno() {
    local errno_name="$1"
    local expected_errno_num="$2"
    local expected_errno_text="$3"

    local actual_errno_num
    local actual_errno_text

    actual_errno_num=$(errno "$errno_name")
    actual_errno_text=$(errno "$errno_name" 2>&1)

    assert_equals "$expected_errno_num" "$actual_errno_num" "Error: Unexpected errno number for $errno_name"
    assert_equals "$expected_errno_text" "$actual_errno_text" "Error: Unexpected errno text for $errno_name"
}

# Call the run_tests function
run_tests
