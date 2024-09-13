#!/usr/bin/env bash

# Check if the function errno is already defined
if ! declare -f errno &> /dev/null; then
    
    # Function: errno
    #
    # Description: This function takes an errno code or errno number and prints the corresponding error message to STDOUT. Sets the exit code to the errno value and returns, unless there is an internal error.
    #
    # Usage: errno [errno_code|errno_number]
    #
    # Example: errno EACCES
    #
    # Returns: "error_code: error_text"
    #
    # Errors: 2, 22
    #   2: Could not find system errno.h
    #  22: Invalid errno name
    #
    function errno() {
        # Usage: errno [errno_code|errno_number]
        if [ -z "$1" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
            echo "Usage: errno [errno_code|errno_number]"
            echo "Example: errno EACCES"
            return 0
        fi

        local errno_code
        errno_code="$(to_upper "$1")"
        local errno_file
        if [ -f "/usr/include/sys/errno.h" ]; then
            errno_file="/usr/include/sys/errno.h"
        elif [ -f "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sys/errno.h" ]; then
            errno_file="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sys/errno.h"
        else
            echo "Error: Could not lookup error code '${errno_code}' system errno.h not found." >&2
            return 2
        fi

        local line errno_num errno_text

        if [[ "$errno_code" =~ ^[0-9]+$ ]]; then
            line=$(grep -wE "#define [A-Z_]*[ \t]*\b$errno_code\b" "$errno_file")
            errno_code=$(echo "$line" | awk '{print $2}')
        else
            line=$(grep -wE "#define $errno_code[ \t]*" "$errno_file")
        fi

        errno_num=$(echo "$line" | awk '{print $3}')
        errno_text=$(echo "$line" | sed -e 's/#define[ \t]*[A-Z0-9_]*[ \t]*[0-9]*[ \t]*\/\* \(.*\) \*\//\1/')

        if [ -z "$errno_num" ]; then
            echo "Error: Invalid errno code $errno_code" >&2
            return 22
        else
            echo "($errno_code: $errno_num): $errno_text"
            return "$errno_num"
        fi
   }

    # Function: to_upper
    #
    # Description: This function converts a string to uppercase
    #
    # Usage: to_upper <string>
    #
    # Example: to_upper "hello"
    #
    # Returns: "HELLO"
    #
    # Errors: None
    #
    function to_upper() {
        local str="$1"
        echo "${str^^}"
    }

    # Function: warn_errno
    #
    # Description: This function prints a warning using the errno function to STDERR and returns the error number
    #
    # Usage: warn <errno_code>
    #
    function errno_warn() {
        echo "WARNING: $(errno "$@")" >&2
        return $?
    }

    # Function: exit_errno
    #
    # Description: This function prints an error to STDERROR using the errno function and exits with the error number
    #
    # Usage: warn <errno_code>
    #
    function errno_exit() {
        echo "ERROR: $(errno "$@")" >&2
        exit $?
    }

fi