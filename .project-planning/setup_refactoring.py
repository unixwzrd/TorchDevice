#!/usr/bin/env python3
"""
Setup script for TorchDevice refactoring.

This script creates the directory structure and empty files for the refactored TorchDevice package.
"""

import os
import shutil
import sys

# Define the directory structure
PACKAGE_STRUCTURE = {
    "TorchDevice": {
        "__init__.py": "",
        "core.py": "",
        "logging.py": "",
        "utils.py": "",
        "cuda": {
            "__init__.py": "",
            "mocks.py": "",
            "events.py": "",
            "streams.py": "",
        },
        "mps": {
            "__init__.py": "",
            "mocks.py": "",
            "events.py": "",
            "streams.py": "",
        },
        "tensor": {
            "__init__.py": "",
            "operations.py": "",
        },
        "module": {
            "__init__.py": "",
            "operations.py": "",
        },
    }
}

def create_directory_structure(base_dir, structure, parent_path=""):
    """
    Create the directory structure recursively.
    
    Args:
        base_dir (str): The base directory.
        structure (dict): The directory structure.
        parent_path (str, optional): The parent path. Defaults to "".
    """
    for name, content in structure.items():
        path = os.path.join(base_dir, parent_path, name)
        
        if isinstance(content, dict):
            # Create directory
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
            
            # Recursively create subdirectories and files
            create_directory_structure(base_dir, content, os.path.join(parent_path, name))
        else:
            # Create file
            with open(path, "w") as f:
                f.write(content)
            print(f"Created file: {path}")

def backup_original_file(original_file):
    """
    Backup the original TorchDevice.py file.
    
    Args:
        original_file (str): The path to the original file.
    
    Returns:
        str: The path to the backup file.
    """
    backup_file = original_file + ".bak"
    shutil.copy2(original_file, backup_file)
    print(f"Backed up original file to: {backup_file}")
    return backup_file

def main():
    """
    Main function.
    """
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if TorchDevice.py exists
    original_file = os.path.join(base_dir, "TorchDevice", "TorchDevice.py")
    if not os.path.isfile(original_file):
        print(f"Error: Original file not found: {original_file}")
        sys.exit(1)
    
    # Backup the original file
    backup_file = backup_original_file(original_file)
    
    # Create the directory structure
    create_directory_structure(base_dir, PACKAGE_STRUCTURE)
    
    print("\nSetup complete!")
    print(f"Original file backed up to: {backup_file}")
    print("Directory structure created.")
    print("\nNext steps:")
    print("1. Implement the logging module first (see logging_module_implementation.md)")
    print("2. Implement the utils module (see initial_package_implementation.md)")
    print("3. Implement the events and streams modules (see events_streams_implementation.md)")
    print("4. Implement the tensor and module operations modules (see tensor_operations_implementation.md and module_operations_implementation.md)")
    print("5. Implement the core module (see initial_package_implementation.md)")
    print("6. Run tests to ensure functionality is preserved (tests/run_tests_and_install.py)")

if __name__ == "__main__":
    main()
